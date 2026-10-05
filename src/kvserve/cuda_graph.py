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

        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.outputs: dict[int, torch.Tensor] = {}

    def _metadata(self, n: int) -> AttentionMetadata:
        return AttentionMetadata(
            slot_mapping=self.slot_mapping[:n],
            block_tables=self.block_tables[:n],
            seq_lens=self.seq_lens[:n],
            query_lens=self.query_lens[:n],
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


def piecewise_sizes(max_tokens: int) -> list[int]:
    """Token-count buckets: fine-grained where decode-heavy mixed steps land, coarser for
    large prefill steps (padding a 1100-token step to 1280 costs <= ~16% extra GEMM work)."""
    sizes = [1, 2, 4, 8]
    sizes += list(range(16, 257, 16))
    sizes += list(range(320, 1025, 64))
    sizes += list(range(1280, max_tokens + 1, 256))
    sizes = [s for s in sizes if s <= max_tokens]
    if sizes[-1] != max_tokens:
        sizes.append(max_tokens)
    return sizes


class PiecewiseGraphRunner:
    """CUDA graphs around everything except attention, for any step shape.

    Decode-only graphs (DecodeGraphRunner) cannot serve steps that mix prefill chunks and
    decodes: attention's launch grid and metadata depend on how tokens split into
    sequences. Everything else in the model is row-wise over the flat token batch, so it
    depends only on the token count. The forward pass is split at the attention calls
    into num_layers + 1 pieces (LlamaForCausalLM.piece); each piece is captured once per
    token-count bucket, and attention runs eagerly in between on the real (unpadded)
    tokens with the step's own metadata.

    Pieces never read or write the KV cache, so padding rows are harmless and capture
    warm-up needs no scratch space. A step costs num_layers + 1 graph replays plus
    num_layers attention launches instead of ~300 individual kernel launches.
    """

    def __init__(self, model: LlamaForCausalLM, max_tokens: int):
        self.model = model
        self.sizes = piecewise_sizes(max_tokens)
        self.max_tokens = self.sizes[-1]
        cfg = model.cfg
        p = next(model.parameters())
        dev, dtype = p.device, p.dtype
        t = self.max_tokens
        self.input_ids = torch.zeros(t, dtype=torch.long, device=dev)
        self.positions = torch.zeros(t, dtype=torch.long, device=dev)
        self.x = torch.zeros(t, cfg.hidden_size, dtype=dtype, device=dev)
        self.q = torch.zeros(t, cfg.num_heads, cfg.head_dim, dtype=dtype, device=dev)
        self.k = torch.zeros(t, cfg.num_kv_heads, cfg.head_dim, dtype=dtype, device=dev)
        self.v = torch.zeros(t, cfg.num_kv_heads, cfg.head_dim, dtype=dtype, device=dev)
        self.attn = torch.zeros(t, cfg.num_heads, cfg.head_dim, dtype=dtype, device=dev)
        self.hidden = torch.zeros(t, cfg.hidden_size, dtype=dtype, device=dev)
        self.num_pieces = cfg.num_layers + 1
        self.graphs: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}

    def _run_piece(self, i: int, n: int) -> None:
        """Piece i on the first n rows of the static buffers, writing results back into them."""
        x, q, k, v = self.model.piece(i, self.input_ids[:n], self.positions[:n], self.x[:n], self.attn[:n])
        if q is None:
            self.hidden[:n].copy_(x)
            return
        self.x[:n].copy_(x)
        self.q[:n].copy_(q)
        self.k[:n].copy_(k)
        self.v[:n].copy_(v)

    @torch.inference_mode()
    def capture(self) -> None:
        pool = None
        stream = torch.cuda.Stream()
        for n in reversed(self.sizes):
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    for i in range(self.num_pieces):
                        self._run_piece(i, n)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            for i in range(self.num_pieces):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool):
                    self._run_piece(i, n)
                pool = graph.pool()
                self.graphs[(i, n)] = graph
        torch.cuda.synchronize()

    def bucket(self, n: int) -> int | None:
        for size in self.sizes:
            if size >= n:
                return size
        return None

    @torch.inference_mode()
    def run(
        self, input_ids: list[int], positions: list[int], kv_caches: torch.Tensor, meta: AttentionMetadata
    ) -> torch.Tensor:
        n = len(input_ids)
        size = self.bucket(n)
        assert size is not None, f"{n} tokens exceed largest piecewise bucket {self.max_tokens}"
        # Rows n..size keep stale but valid ids/positions from earlier steps; their
        # results are discarded and they never reach attention or the KV cache.
        self.input_ids[:n].copy_(torch.tensor(input_ids), non_blocking=True)
        self.positions[:n].copy_(torch.tensor(positions), non_blocking=True)
        layers = self.model.layers
        for i in range(self.num_pieces):
            self.graphs[(i, size)].replay()
            if i < len(layers):
                layers[i].self_attn.attend(self.q[:n], self.k[:n], self.v[:n], kv_caches[i], meta, out=self.attn[:n])
        return self.hidden[:n]
