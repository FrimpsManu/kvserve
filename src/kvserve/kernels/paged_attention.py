"""Triton paged attention for mixed prefill/decode batches.

One kernel serves every sequence in the flat batch, whatever its query length.

Grid: (sequence, kv_head, query tile). A program owns BLOCK_Q query tokens of one
sequence and *all* GROUP query heads that share one KV head (GQA), laid out as
BLOCK_Q * GROUP rows of a single [M, D] tile. It walks the sequence's block table and
streams each KV page from HBM exactly once, so every K/V byte loaded is reused by the
GROUP heads; the reference implementation instead materialises a padded copy of every
sequence's context per step. Softmax is computed online (FlashAttention style) with
fp32 statistics, so no [M, S] score matrix is ever stored.

Decode is memory bound: per step it must read the whole KV context once. The figure
of merit for this kernel is therefore achieved HBM bandwidth (see bench/bench_kernel.py).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_attention_kernel(
    Q,  # [T, H, D]
    K_CACHE,  # [num_blocks, BLOCK_SIZE, H_KV, D]
    V_CACHE,
    OUT,  # [T, H, D]
    BLOCK_TABLES,  # [B, max_blocks] int32/int64
    SEQ_LENS,  # [B]
    QUERY_START_LOC,  # [B + 1]
    scale,
    stride_qt, stride_qh,
    stride_kb, stride_ks, stride_kh,
    stride_vb, stride_vs, stride_vh,
    stride_ot, stride_oh,
    stride_bt,
    GROUP: tl.constexpr,  # query heads per kv head
    BLOCK_Q: tl.constexpr,  # query tokens per program
    BLOCK_SIZE: tl.constexpr,  # tokens per KV page
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_PAD: tl.constexpr,
    IEEE: tl.constexpr,  # exact fp32 dots (no TF32); used for fp32 correctness runs
):  # fmt: skip
    seq = tl.program_id(0)
    kv_head = tl.program_id(1)
    q_tile = tl.program_id(2)

    q_start = tl.load(QUERY_START_LOC + seq)
    q_len = tl.load(QUERY_START_LOC + seq + 1) - q_start
    if q_tile * BLOCK_Q >= q_len:
        return
    seq_len = tl.load(SEQ_LENS + seq)
    context_len = seq_len - q_len  # tokens already cached before this step

    M: tl.constexpr = BLOCK_Q * GROUP
    rows = tl.arange(0, M)
    q_idx = q_tile * BLOCK_Q + rows // GROUP  # token index within this sequence's query
    head = kv_head * GROUP + rows % GROUP
    row_valid = q_idx < q_len
    q_pos = context_len + q_idx  # absolute position, for the causal mask

    d = tl.arange(0, HEAD_DIM_PAD)
    d_valid = d < HEAD_DIM
    q_ptrs = Q + (q_start + q_idx)[:, None] * stride_qt + head[:, None] * stride_qh + d[None, :]
    q = tl.load(q_ptrs, mask=row_valid[:, None] & d_valid[None, :], other=0.0)

    m_i = tl.full([M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([M], dtype=tl.float32)
    acc = tl.zeros([M, HEAD_DIM_PAD], dtype=tl.float32)

    # Causal: the last query row of this tile sees keys up to its own position.
    last_q = tl.minimum((q_tile + 1) * BLOCK_Q, q_len) - 1
    kv_end = context_len + last_q + 1
    offs_s = tl.arange(0, BLOCK_SIZE)
    qk_scale = scale * 1.4426950408889634  # fold log2(e) in so we can use exp2

    for start in range(0, kv_end, BLOCK_SIZE):
        block = tl.load(BLOCK_TABLES + seq * stride_bt + start // BLOCK_SIZE).to(tl.int64)
        k_pos = start + offs_s
        kv_mask = (k_pos < kv_end)[:, None] & d_valid[None, :]
        k = tl.load(
            K_CACHE + block * stride_kb + offs_s[:, None] * stride_ks + kv_head * stride_kh + d[None, :],
            mask=kv_mask, other=0.0,
        )  # fmt: skip
        v = tl.load(
            V_CACHE + block * stride_vb + offs_s[:, None] * stride_vs + kv_head * stride_vh + d[None, :],
            mask=kv_mask, other=0.0,
        )  # fmt: skip

        if IEEE:  # noqa: SIM108 (compile-time branch inside a Triton kernel)
            s = tl.dot(q, tl.trans(k), input_precision="ieee")
        else:
            s = tl.dot(q, tl.trans(k))
        s = s * qk_scale  # [M, BLOCK_SIZE]
        causal = k_pos[None, :] <= q_pos[:, None]
        s = tl.where(causal, s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        # Rows whose keys are all masked so far keep m = -inf; avoid (-inf) - (-inf).
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.math.exp2(s - m_safe[:, None])
        alpha = tl.math.exp2(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        if IEEE:  # noqa: SIM108 (compile-time branch inside a Triton kernel)
            pv = tl.dot(p.to(v.dtype), v, input_precision="ieee")
        else:
            pv = tl.dot(p.to(v.dtype), v)
        acc = acc * alpha[:, None] + pv
        m_i = m_new

    out = acc / tl.where(l_i == 0, 1.0, l_i)[:, None]
    o_ptrs = OUT + (q_start + q_idx)[:, None] * stride_ot + head[:, None] * stride_oh + d[None, :]
    tl.store(o_ptrs, out.to(OUT.dtype.element_ty), mask=row_valid[:, None] & d_valid[None, :])


def paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    max_query_len: int,
    scale: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """q: [T, H, D]; caches: [num_blocks, block_size, H_kv, D]. Returns [T, H, D] (in `out` if given)."""
    num_tokens, num_heads, head_dim = q.shape
    _, block_size, num_kv_heads, _ = k_cache.shape
    assert num_heads % num_kv_heads == 0
    assert k_cache.stride(-1) == 1 and q.stride(-1) == 1
    group = num_heads // num_kv_heads
    num_seqs = seq_lens.shape[0]

    # tl.dot needs M >= 16. Decode-only batches use the smallest tile (wasted rows are
    # masked); prefill uses 64-128 rows so each KV page is reused across many queries.
    min_q = max(1, 16 // group)
    block_q = min_q if max_query_len <= min_q else max(min_q, 64 // group)

    if out is None:
        out = torch.empty_like(q)
    assert out.shape == q.shape and out.stride(-1) == 1
    grid = (num_seqs, num_kv_heads, triton.cdiv(max_query_len, block_q))
    _paged_attention_kernel[grid](
        q, k_cache, v_cache, out, block_tables, seq_lens, query_start_loc, scale,
        q.stride(0), q.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        out.stride(0), out.stride(1),
        block_tables.stride(0),
        GROUP=group, BLOCK_Q=block_q, BLOCK_SIZE=block_size,
        HEAD_DIM=head_dim, HEAD_DIM_PAD=triton.next_power_of_2(head_dim), IEEE=q.dtype == torch.float32,
        num_warps=4, num_stages=2,
    )  # fmt: skip
    return out
