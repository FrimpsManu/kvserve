"""Owns the model and the KV pool; turns a SchedulerOutput into one forward pass."""

from __future__ import annotations

import torch

from kvserve.attention import AttentionMetadata, get_backend
from kvserve.config import EngineConfig, ModelConfig, resolve_model_path
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
        # [layers, k/v, blocks, block_size, kv_heads, head_dim]
        self.kv_caches = torch.zeros(
            mc.num_layers, 2, self.num_kv_blocks, config.block_size, mc.num_kv_heads, mc.head_dim,
            dtype=self.dtype, device=self.device,
        )  # fmt: skip
        self.generator = torch.Generator().manual_seed(config.seed)

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
            slots += (kv.slot(seq, p) for p in range(start, start + n))
            seq_lens.append(start + n)
            query_lens.append(n)
            if start + n == seq.num_tokens:  # chunked prefill not done yet -> no sample
                sample_rows.append(len(input_ids) - 1)
                sample_seqs.append(seq)

        meta = self._build_metadata(sched, slots, seq_lens, query_lens)
        dev = self.device
        hidden = self.model(
            torch.tensor(input_ids, device=dev), torch.tensor(positions, device=dev), self.kv_caches, meta
        )
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
        q_gather, q_valid, offset = [], [], 0
        query_start_loc = [0]
        for n in query_lens:
            q_gather.append(list(range(offset, offset + n)) + [0] * (max_q - n))
            q_valid.append([True] * n + [False] * (max_q - n))
            offset += n
            query_start_loc.append(offset)
        return AttentionMetadata(
            slot_mapping=torch.tensor(slots, dtype=torch.long, device=dev),
            block_tables=torch.tensor(tables, dtype=torch.long, device=dev),
            seq_lens=torch.tensor(seq_lens, device=dev),
            query_lens=torch.tensor(query_lens, device=dev),
            q_gather=torch.tensor(q_gather, dtype=torch.long, device=dev),
            q_valid=torch.tensor(q_valid, device=dev),
            query_start_loc=torch.tensor(query_start_loc, device=dev),
            max_seq_len=max(seq_lens),
            max_query_len=max_q,
        )
