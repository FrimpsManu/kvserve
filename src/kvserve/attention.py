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
    query_start_loc: torch.Tensor  # [B + 1] offset of each sequence's first token in the flat batch
    max_seq_len: int
    max_query_len: int
    # Padded layout, needed only by the torch reference backend.
    q_gather: torch.Tensor | None = None  # [B, max_q] flat index of each query token, 0-padded
    q_valid: torch.Tensor | None = None  # [B, max_q] bool, False on padding


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
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        meta: AttentionMetadata,
        scale: float,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert meta.q_gather is not None and meta.q_valid is not None, "torch backend needs padded-layout metadata"
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

        res = F.scaled_dot_product_attention(qp, k, v, attn_mask=mask[:, None], scale=scale)
        res = res.transpose(1, 2)[meta.q_valid]  # [T, H, D] in flat token order
        assert res.shape[0] == num_tokens and num_seqs == meta.q_valid.shape[0]
        return out.copy_(res) if out is not None else res


class TritonAttention(TorchAttention):
    """Paged attention via the Triton kernel (NVIDIA/AMD GPUs). K/V writes reuse the torch path."""

    @staticmethod
    def forward(
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        meta: AttentionMetadata,
        scale: float,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from kvserve.kernels.paged_attention import paged_attention

        return paged_attention(
            q, k_cache, v_cache, meta.block_tables, meta.seq_lens, meta.query_start_loc, meta.max_query_len, scale,
            out=out,
        )  # fmt: skip


def triton_available() -> bool:
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available()


def get_backend(name: str, device: str, block_size: int) -> type[TorchAttention]:
    if name == "auto":
        name = "triton" if device.startswith("cuda") and triton_available() and block_size >= 16 else "torch"
    if name == "torch":
        return TorchAttention
    if name == "triton":
        if not triton_available():
            raise RuntimeError("triton backend needs a CUDA GPU and the triton package")
        if block_size < 16:
            raise ValueError("triton backend needs block_size >= 16 (tl.dot minimum tile)")
        return TritonAttention
    raise ValueError(f"unknown attention backend {name!r}")
