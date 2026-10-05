"""Continuous-batching scheduler with chunked prefill and recompute preemption.

Every step gets a token budget (`max_num_batched_tokens`). Running sequences are
served first (in arrival order), so in-flight decodes are never starved by new
prompts; leftover budget admits waiting requests, whose prompts may be split across
several steps (chunked prefill). There is no separate "prefill phase": a sequence
simply needs `num_tokens - num_computed_tokens` more tokens computed.

If the KV pool runs out, the most recently admitted running sequence is preempted:
its blocks are freed and it goes back to the front of the waiting queue, to be
recomputed later (cheap with prefix caching, since its full blocks stay cached
until evicted).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from kvserve.kv_cache import KVCacheManager
from kvserve.sequence import Sequence, SequenceStatus


@dataclass
class SchedulerOutput:
    scheduled: list[tuple[Sequence, int]] = field(default_factory=list)  # (seq, tokens this step)
    num_preempted: int = 0

    @property
    def num_tokens(self) -> int:
        return sum(n for _, n in self.scheduled)

    @property
    def is_empty(self) -> bool:
        return not self.scheduled


class Scheduler:
    def __init__(self, kv: KVCacheManager, max_num_seqs: int, max_num_batched_tokens: int):
        self.kv = kv
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []

    def add(self, seq: Sequence) -> None:
        max_tokens = self.kv.num_blocks * self.kv.block_size
        if seq.num_prompt_tokens + 1 > max_tokens:
            raise ValueError(f"prompt of {seq.num_prompt_tokens} tokens can never fit in the KV cache")
        self.waiting.append(seq)

    @property
    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    def schedule(self) -> SchedulerOutput:
        out = SchedulerOutput()
        budget = self.max_num_batched_tokens

        # 1. Running sequences, oldest first.
        i = 0
        while i < len(self.running) and budget > 0:
            seq = self.running[i]
            n = min(seq.num_tokens - seq.num_computed_tokens, budget)
            while not self.kv.allocate_slots(seq, n):
                victim = self.running.pop()  # newest admitted
                self._preempt(victim)
                out.num_preempted += 1
                if victim is seq:
                    break
            else:
                out.scheduled.append((seq, n))
                budget -= n
                i += 1

        # 2. Admit waiting sequences, unless we just had to preempt (memory is tight).
        while self.waiting and budget > 0 and not out.num_preempted and len(self.running) < self.max_num_seqs:
            seq = self.waiting[0]
            cached = self.kv.find_cached_prefix(seq)
            n = min(seq.num_tokens - seq.num_computed_tokens - len(cached) * self.kv.block_size, budget)
            if not self.kv.allocate_slots(seq, n, cached):
                break
            self.waiting.popleft()
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            out.scheduled.append((seq, n))
            budget -= n

        return out

    def _preempt(self, seq: Sequence) -> None:
        self.kv.free(seq)
        seq.num_computed_tokens = 0
        seq.status = SequenceStatus.WAITING
        seq.num_preemptions += 1
        self.waiting.appendleft(seq)

    def finish(self, seq: Sequence) -> None:
        """Release a finished or aborted sequence."""
        if seq in self.running:
            self.running.remove(seq)
        elif seq in self.waiting:
            self.waiting.remove(seq)
        self.kv.free(seq)
        seq.status = SequenceStatus.FINISHED
