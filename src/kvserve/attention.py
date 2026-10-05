"""Attention over the paged KV cache.

All sequences in a step are packed into one flat token dimension. `AttentionMetadata`
describes how those tokens split into sequences and where each token's K/V lives in
the block pool. Backends implement `write_kv` and `forward`; `TorchAttention` is the
portable reference (CPU/MPS/CUDA). Optimised kernels plug in behind the same interface.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class AttentionMetadata:
    slot_mapping: torch.Tensor  # [T] flat KV slot for each new token
    block_tables: torch.Tensor  # [B, max_blocks] physical block ids, 0-padded
    seq_lens: torch.Tensor  # [B] context length including this step's tokens
    query_lens: torch.Tensor  # [B] tokens computed this step
    q_gather: torch.Tensor  # [B, max_q] flat index of each query token, 0-padded
    q_valid: torch.Tensor  # [B, max_q] bool, False on padding
    max_seq_len: int
    max_query_len: int


class TorchAttention:
    """Reference paged attention built from PyTorch ops.

    Gathers each sequence's K/V blocks into a padded dense tensor and runs SDPA with a
    causal mask offset by the cached context length. Correct for any mix of prefill
    chunks and decode tokens; memory traffic is not optimal (that is the kernel's job).
    """

    @staticmethod
    def write_kv(
        k_cache: torch.Tensor, v_cache: torch.Tensor, k: torch.Tensor, v: torch.Tensor, slots: torch.Tensor
    ) -> None:
        # caches: [num_blocks, block_size, kv_heads, head_dim]; k, v: [T, kv_heads, head_dim]
        k_cache.view(-1, *k_cache.shape[2:])[slots] = k
        v_cache.view(-1, *v_cache.shape[2:])[slots] = v

    @staticmethod
    def forward(
        q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, meta: AttentionMetadata, scale: float
    ) -> torch.Tensor:
        num_tokens, num_heads, head_dim = q.shape
        block_size, num_kv_heads = k_cache.shape[1], k_cache.shape[2]
        num_seqs = meta.block_tables.shape[0]

        num_blocks = -(-meta.max_seq_len // block_size)
        tables = meta.block_tables[:, :num_blocks]
        # [B, S, kv_heads, D] -> [B, kv_heads, S, D]
        k = k_cache[tables].flatten(1, 2)[:, : meta.max_seq_len].transpose(1, 2)
        v = v_cache[tables].flatten(1, 2)[:, : meta.max_seq_len].transpose(1, 2)
        if num_kv_heads != num_heads:  # grouped-query attention
            k = k.repeat_interleave(num_heads // num_kv_heads, dim=1)
            v = v.repeat_interleave(num_heads // num_kv_heads, dim=1)

        qp = q[meta.q_gather].transpose(1, 2)  # [B, H, max_q, D]

        # Query i of a sequence sits at absolute position (seq_len - query_len + i) and
        # may attend to every key at or before that position.
        device = q.device
        q_pos = (meta.seq_lens - meta.query_lens)[:, None] + torch.arange(meta.max_query_len, device=device)
        k_pos = torch.arange(meta.max_seq_len, device=device)
        mask = k_pos[None, None, :] <= q_pos[:, :, None]  # [B, max_q, S]

        out = F.scaled_dot_product_attention(qp, k, v, attn_mask=mask[:, None], scale=scale)
        out = out.transpose(1, 2)[meta.q_valid]  # [T, H, D] in flat token order
        assert out.shape[0] == num_tokens and num_seqs == meta.q_valid.shape[0]
        return out
