"""Request state tracked by the scheduler."""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 1.0  # 0 means greedy
    top_p: float = 1.0
    top_k: int = -1  # -1 disables
    max_tokens: int = 256
    stop_token_ids: tuple[int, ...] = ()
    ignore_eos: bool = False

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")


class SequenceStatus(enum.Enum):
    WAITING = enum.auto()
    RUNNING = enum.auto()
    FINISHED = enum.auto()


class FinishReason(enum.StrEnum):
    STOP = "stop"  # eos or stop token
    LENGTH = "length"  # max_tokens or max_model_len
    ABORT = "abort"


class Sequence:
    """One generation request.

    `num_computed_tokens` counts tokens whose K/V are already in the cache. A sequence
    needs computing while `num_computed_tokens < num_tokens`; this single invariant
    covers prefill, chunked prefill, decode and recompute after preemption.
    """

    def __init__(self, request_id: str, prompt_token_ids: list[int], params: SamplingParams):
        if not prompt_token_ids:
            raise ValueError("prompt must contain at least one token")
        self.request_id = request_id
        self.prompt_token_ids = list(prompt_token_ids)
        self.output_token_ids: list[int] = []
        self.params = params
        self.status = SequenceStatus.WAITING
        self.finish_reason: FinishReason | None = None

        self.block_table: list[int] = []
        self.num_computed_tokens = 0
        self.num_cached_prompt_tokens = 0  # prefix-cache hits at admission
        self.block_hashes: list[bytes] = []  # hashes of this sequence's full blocks
        self.num_preemptions = 0
        # Speculative decoding: draft tokens to verify in the next step (not yet part of
        # the sequence), and lifetime counts for the acceptance rate.
        self.spec_token_ids: list[int] = []
        self.num_draft_tokens = 0
        self.num_accepted_tokens = 0

        self.arrival_time = time.perf_counter()
        self.first_token_time: float | None = None
        self.finish_time: float | None = None

    @property
    def token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.output_token_ids

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def is_finished(self) -> bool:
        return self.status is SequenceStatus.FINISHED

    def append_token(self, token_id: int) -> None:
        if self.first_token_time is None:
            self.first_token_time = time.perf_counter()
        self.output_token_ids.append(token_id)

    def __repr__(self) -> str:
        return (
            f"Sequence({self.request_id!r}, tokens={self.num_tokens}, "
            f"computed={self.num_computed_tokens}, status={self.status.name})"
        )
