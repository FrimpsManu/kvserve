"""Synchronous engine: request intake, the schedule -> execute -> update loop, outputs."""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass

from transformers import AutoTokenizer

from kvserve.config import EngineConfig
from kvserve.kv_cache import KVCacheManager
from kvserve.model_runner import ModelRunner
from kvserve.scheduler import Scheduler
from kvserve.sequence import FinishReason, SamplingParams, Sequence


@dataclass
class RequestOutput:
    request_id: str
    new_token_ids: list[int]
    output_token_ids: list[int]
    finished: bool
    finish_reason: FinishReason | None
    num_prompt_tokens: int
    num_cached_tokens: int


@dataclass
class StepStats:
    num_tokens: int
    num_seqs: int
    num_preempted: int
    duration_s: float
    kv_usage: float


class LLMEngine:
    def __init__(self, config: EngineConfig | None = None):
        self.config = config or EngineConfig()
        self.runner = ModelRunner(self.config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.runner.model_path)
        self.kv = KVCacheManager(self.runner.num_kv_blocks, self.config.block_size, self.config.enable_prefix_caching)
        self.scheduler = Scheduler(self.kv, self.config.max_num_seqs, self.config.max_num_batched_tokens)
        self.eos_token_ids = set(self.runner.model_config.eos_token_ids)
        self.requests: dict[str, Sequence] = {}
        self.last_step: StepStats | None = None
        self._ids = itertools.count()

    def add_request(
        self, prompt: str | list[int], params: SamplingParams | None = None, request_id: str | None = None
    ) -> str:
        params = params or SamplingParams()
        request_id = request_id or f"req-{next(self._ids)}"
        if request_id in self.requests:
            raise ValueError(f"duplicate request id {request_id!r}")
        token_ids = self.tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt)
        if len(token_ids) >= self.config.max_model_len:
            raise ValueError(f"prompt has {len(token_ids)} tokens; max_model_len is {self.config.max_model_len}")
        seq = Sequence(request_id, token_ids, params)
        self.scheduler.add(seq)
        self.requests[request_id] = seq
        return request_id

    def abort(self, request_id: str) -> None:
        seq = self.requests.pop(request_id, None)
        if seq is not None and not seq.is_finished:
            seq.finish_reason = FinishReason.ABORT
            self.scheduler.finish(seq)

    def has_unfinished(self) -> bool:
        return self.scheduler.has_unfinished

    def step(self) -> list[RequestOutput]:
        start = time.perf_counter()
        self.last_step = None
        sched = self.scheduler.schedule()
        if sched.is_empty:
            return []
        sampled = self.runner.execute(sched, self.kv)

        outputs = []
        for seq, n in sched.scheduled:
            seq.num_computed_tokens += n
            self.kv.cache_full_blocks(seq)
            token = sampled.get(seq)
            if token is None:
                continue  # mid-prefill chunk
            seq.append_token(token)
            reason = self._check_stop(seq, token)
            if reason is not None:
                seq.finish_reason = reason
                seq.finish_time = time.perf_counter()
                self.scheduler.finish(seq)
                del self.requests[seq.request_id]
            outputs.append(
                RequestOutput(
                    seq.request_id,
                    [token],
                    seq.output_token_ids,
                    reason is not None,
                    reason,
                    seq.num_prompt_tokens,
                    seq.num_cached_prompt_tokens,
                )  # fmt: skip
            )
        self.last_step = StepStats(
            sched.num_tokens, len(sched.scheduled), sched.num_preempted, time.perf_counter() - start, self.kv.usage
        )
        return outputs

    def _check_stop(self, seq: Sequence, token: int) -> FinishReason | None:
        p = seq.params
        if (not p.ignore_eos and token in self.eos_token_ids) or token in p.stop_token_ids:
            return FinishReason.STOP
        if len(seq.output_token_ids) >= p.max_tokens or seq.num_tokens >= self.config.max_model_len:
            return FinishReason.LENGTH
        return None

    def generate(self, prompts: list[str] | list[list[int]], params: SamplingParams | None = None) -> list[list[int]]:
        """Run prompts to completion; returns output token ids in prompt order."""
        ids = [self.add_request(p, params) for p in prompts]
        results: dict[str, list[int]] = {}
        while self.has_unfinished():
            for out in self.step():
                if out.finished:
                    results[out.request_id] = out.output_token_ids
        return [results[i] for i in ids]
