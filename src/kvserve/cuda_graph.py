"""CUDA Graph capture and replay for decode steps.

Decode is launch bound: a step issues ~300 small kernels from Python, and for a 1B
model the CPU takes ~10 ms to issue work the GPU finishes in ~2.5 ms. A CUDA graph
records the whole forward pass once and replays it with a single launch.

A graph replays fixed kernels on fixed memory addresses, so:
- every input lives in a static buffer that is overwritten before each replay;
- one graph is captured per padded batch size ("bucket"); a batch of n runs the
  smallest bucket >= n, padding the extra rows;
- padding rows write their K/V to a scratch block reserved outside the allocator's
  pool, so they never touch cache contents that real (or prefix-cached) sequences use;
- only uniform decode steps (one new token per sequence) are captured; prefill and
  mixed steps run eagerly.

Graphs require the Triton attention backend: the torch reference derives tensor
shapes from per-step Python ints, which a graph cannot vary.
"""

from __future__ import annotations

import torch

from kvserve.attention import AttentionMetadata
from kvserve.model import LlamaForCausalLM


def capture_sizes(max_batch: int) -> list[int]:
    sizes = [s for s in (1, 2, 4, 8) if s <= max_batch]
    sizes += list(range(16, max_batch + 1, 16))
    if sizes[-1] != max_batch:
        sizes.append(max_batch)
    return sizes


class DecodeGraphRunner:
    def __init__(
        self,
        model: LlamaForCausalLM,
        kv_caches: torch.Tensor,
        scratch_block: int,
        block_size: int,
        max_batch: int,
        max_blocks_per_seq: int,
    ):
        self.model = model
        self.kv_caches = kv_caches
        self.scratch_block = scratch_block
        self.scratch_slot = scratch_block * block_size
        self.sizes = capture_sizes(max_batch)
        self.max_batch = self.sizes[-1]
        dev = kv_caches.device

        b = self.max_batch
        self.input_ids = torch.zeros(b, dtype=torch.long, device=dev)
        self.positions = torch.zeros(b, dtype=torch.long, device=dev)
        self.slot_mapping = torch.full((b,), self.scratch_slot, dtype=torch.long, device=dev)
        self.block_tables = torch.full((b, max_blocks_per_seq), scratch_block, dtype=torch.long, device=dev)
        self.seq_lens = torch.ones(b, dtype=torch.long, device=dev)
        # Every sequence contributes exactly one query token, so these never change.
        self.query_lens = torch.ones(b, dtype=torch.long, device=dev)
        self.query_start_loc = torch.arange(b + 1, dtype=torch.long, device=dev)
        self._unused = torch.zeros(b, 1, dtype=torch.long, device=dev)  # torch-backend-only fields

        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.outputs: dict[int, torch.Tensor] = {}

    def _metadata(self, n: int) -> AttentionMetadata:
        return AttentionMetadata(
            slot_mapping=self.slot_mapping[:n],
            block_tables=self.block_tables[:n],
            seq_lens=self.seq_lens[:n],
            query_lens=self.query_lens[:n],
            q_gather=self._unused[:n],
            q_valid=self._unused[:n].bool(),
            query_start_loc=self.query_start_loc[: n + 1],
            max_seq_len=1,  # unused by the Triton backend
            max_query_len=1,
        )

    @torch.inference_mode()
    def capture(self) -> None:
        """Capture all buckets, largest first so smaller graphs reuse its memory pool.

        Buffers hold scratch-only values during capture, so warm-up runs write K/V only
        to the scratch block.
        """
        pool = None
        stream = torch.cuda.Stream()
        for n in reversed(self.sizes):
            meta = self._metadata(n)
            args = (self.input_ids[:n], self.positions[:n], self.kv_caches, meta)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):  # compile Triton kernels, settle cuBLAS workspaces
                    self.model(*args)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                self.outputs[n] = self.model(*args)
            pool = graph.pool()
            self.graphs[n] = graph
        torch.cuda.synchronize()

    def bucket(self, n: int) -> int | None:
        for size in self.sizes:
            if size >= n:
                return size
        return None

    @torch.inference_mode()
    def run(
        self,
        input_ids: list[int],
        positions: list[int],
        slots: list[int],
        block_tables: list[list[int]],
        seq_lens: list[int],
    ) -> torch.Tensor:
        """Replay the graph for a uniform decode batch; returns hidden states [n, hidden]."""
        n = len(input_ids)
        size = self.bucket(n)
        assert size is not None, f"batch {n} exceeds largest captured graph {self.max_batch}"
        pad = size - n

        self.input_ids[:size].copy_(torch.tensor(input_ids + [0] * pad), non_blocking=True)
        self.positions[:size].copy_(torch.tensor(positions + [0] * pad), non_blocking=True)
        self.slot_mapping[:size].copy_(torch.tensor(slots + [self.scratch_slot] * pad), non_blocking=True)
        self.seq_lens[:size].copy_(torch.tensor(seq_lens + [1] * pad), non_blocking=True)
        # The kernel reads only the first ceil(seq_len / block_size) entries of each row,
        # so only that many columns need refreshing; padding rows point at scratch.
        width = max(len(t) for t in block_tables)
        rows = [t + [self.scratch_block] * (width - len(t)) for t in block_tables]
        rows += [[self.scratch_block] * width] * pad
        self.block_tables[:size, :width].copy_(torch.tensor(rows), non_blocking=True)

        self.graphs[size].replay()
        return self.outputs[size][:n]
