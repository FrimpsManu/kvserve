"""Owns the model and the KV pool; turns a SchedulerOutput into one forward pass."""

from __future__ import annotations

from collections import Counter

import torch

from kvserve.attention import AttentionMetadata, TritonAttention, get_backend
from kvserve.config import EngineConfig, ModelConfig, resolve_model_path
from kvserve.cuda_graph import DecodeGraphRunner, PiecewiseGraphRunner
from kvserve.kv_cache import KVCacheManager
from kvserve.model import load_model
from kvserve.sampler import sample
from kvserve.scheduler import SchedulerOutput
from kvserve.sequence import Sequence


class ModelRunner:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = config.torch_dtype
        path = resolve_model_path(config.model)
        self.model_path = path
        self.model_config = ModelConfig.from_dir(path)
        self.attn_backend = get_backend(config.attention_backend, config.device, config.block_size)
        self.model = load_model(
            path, self.model_config, config.device, self.dtype, config.max_model_len, self.attn_backend
        )
        self.num_kv_blocks = config.num_kv_blocks or self._blocks_for_memory(config.kv_cache_memory_gb)
        mc = self.model_config
        # [layers, k/v, blocks, block_size, kv_heads, head_dim]. One extra block past the
        # allocator's pool is scratch space for CUDA-graph padding rows.
        self.scratch_block = self.num_kv_blocks
        self.kv_caches = torch.zeros(
            mc.num_layers, 2, self.num_kv_blocks + 1, config.block_size, mc.num_kv_heads, mc.head_dim,
            dtype=self.dtype, device=self.device,
        )  # fmt: skip
        self.generator = torch.Generator().manual_seed(config.seed)

        self.step_paths: Counter[str] = Counter()  # decode_graph | piecewise | eager
        self.graphs: DecodeGraphRunner | None = None
        self.piecewise: PiecewiseGraphRunner | None = None
        if self.device.type == "cuda" and self.attn_backend is TritonAttention:
            self._warmup_kernels()
        if config.enable_cuda_graphs and self.device.type == "cuda" and self.attn_backend is TritonAttention:
            if config.enable_piecewise_graphs:
                max_tokens = min(config.max_piecewise_tokens, config.max_num_batched_tokens)
                self.piecewise = PiecewiseGraphRunner(self.model, max_tokens)
                self.piecewise.capture()
            self.graphs = DecodeGraphRunner(
                self.model,
                self.kv_caches,
                scratch_block=self.scratch_block,
                block_size=config.block_size,
                max_batch=min(config.max_graph_batch_size, config.max_num_seqs),
                max_blocks_per_seq=-(-config.max_model_len // config.block_size),
            )
            self.graphs.capture()

    @torch.inference_mode()
    def _warmup_kernels(self) -> None:
        """JIT-compile Triton kernel variants before serving.

        Triton compiles each kernel specialisation on first use, which otherwise stalls
        the first real requests by seconds. Runs one prefill-shaped and one decode-shaped
        forward whose K/V writes and reads all hit the scratch block, so the pool the
        allocator hands out is never touched.
        """
        bs, dev = self.config.block_size, self.device
        prefill_len = min(64, self.config.max_num_batched_tokens)
        for query_len, num_seqs in ((prefill_len, 1), (1, 4)):
            num_tokens = query_len * num_seqs
            meta = AttentionMetadata(
                slot_mapping=torch.full((num_tokens,), self.scratch_block * bs, dtype=torch.long, device=dev),
                block_tables=torch.full(
                    (num_seqs, -(-query_len // bs)), self.scratch_block, dtype=torch.long, device=dev
                ),
                seq_lens=torch.full((num_seqs,), query_len, device=dev),
                query_lens=torch.full((num_seqs,), query_len, device=dev),
                query_start_loc=torch.arange(0, num_tokens + 1, query_len, device=dev),
                max_seq_len=query_len,
                max_query_len=query_len,
            )
            ids = torch.zeros(num_tokens, dtype=torch.long, device=dev)
            positions = torch.arange(query_len, device=dev).repeat(num_seqs)
            self.model(ids, positions, self.kv_caches, meta)
        torch.cuda.synchronize()

    def _count(self, path: str) -> None:
        self.step_paths[path] += 1

    def _blocks_for_memory(self, gb: float) -> int:
        mc = self.model_config
        itemsize = torch.empty((), dtype=self.dtype).element_size()
        block_bytes = 2 * mc.num_layers * self.config.block_size * mc.num_kv_heads * mc.head_dim * itemsize
        return max(1, int(gb * 1024**3) // block_bytes)

    def execute(self, sched: SchedulerOutput, kv: KVCacheManager) -> dict[Sequence, int]:
        """Run one step. Returns sampled tokens for sequences whose context is now complete."""
        input_ids: list[int] = []
        positions: list[int] = []
        slots: list[int] = []
        sample_rows: list[int] = []  # flat index of the last token of sequences that sample
        sample_seqs: list[Sequence] = []
        seq_lens: list[int] = []
        query_lens: list[int] = []

        for seq, n in sched.scheduled:
            start = seq.num_computed_tokens
            tokens = seq.token_ids
            input_ids += tokens[start : start + n]
            positions += range(start, start + n)
            slots += kv.slots(seq, start, n)
            seq_lens.append(start + n)
            query_lens.append(n)
            if start + n == seq.num_tokens:  # chunked prefill not done yet -> no sample
                sample_rows.append(len(input_ids) - 1)
                sample_seqs.append(seq)

        dev = self.device
        if self.graphs is not None and max(query_lens) == 1 and len(query_lens) <= self.graphs.max_batch:
            tables = [seq.block_table for seq, _ in sched.scheduled]
            hidden = self.graphs.run(input_ids, positions, slots, tables, seq_lens)
            self._count("decode_graph")
        elif self.piecewise is not None and len(input_ids) <= self.piecewise.max_tokens:
            meta = self._build_metadata(sched, slots, seq_lens, query_lens)
            hidden = self.piecewise.run(input_ids, positions, self.kv_caches, meta)
            self._count("piecewise")
        else:
            meta = self._build_metadata(sched, slots, seq_lens, query_lens)
            hidden = self.model(
                torch.tensor(input_ids, device=dev), torch.tensor(positions, device=dev), self.kv_caches, meta
            )
            self._count("eager")
        if not sample_seqs:
            return {}
        logits = self.model.compute_logits(hidden[torch.tensor(sample_rows, device=dev)])
        tokens = sample(logits, [s.params for s in sample_seqs], self.generator)
        return dict(zip(sample_seqs, tokens, strict=True))

    def _build_metadata(
        self, sched: SchedulerOutput, slots: list[int], seq_lens: list[int], query_lens: list[int]
    ) -> AttentionMetadata:
        dev = self.device
        max_blocks = max(len(seq.block_table) for seq, _ in sched.scheduled)
        tables = [seq.block_table + [0] * (max_blocks - len(seq.block_table)) for seq, _ in sched.scheduled]
        max_q = max(query_lens)
        query_start_loc = [0]
        for n in query_lens:
            query_start_loc.append(query_start_loc[-1] + n)
        meta = AttentionMetadata(
            slot_mapping=torch.tensor(slots, dtype=torch.long, device=dev),
            block_tables=torch.tensor(tables, dtype=torch.long, device=dev),
            seq_lens=torch.tensor(seq_lens, device=dev),
            query_lens=torch.tensor(query_lens, device=dev),
            query_start_loc=torch.tensor(query_start_loc, device=dev),
            max_seq_len=max(seq_lens),
            max_query_len=max_q,
        )
        if self.attn_backend is not TritonAttention:  # padded layout for the torch reference only
            q_gather, q_valid = [], []
            for start, n in zip(query_start_loc, query_lens, strict=False):
                q_gather.append(list(range(start, start + n)) + [0] * (max_q - n))
                q_valid.append([True] * n + [False] * (max_q - n))
            meta.q_gather = torch.tensor(q_gather, dtype=torch.long, device=dev)
            meta.q_valid = torch.tensor(q_valid, device=dev)
        return meta
