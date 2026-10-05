"""Triton paged attention vs. the torch reference on random paged layouts (CUDA only)."""

import random

import pytest
import torch

from kvserve.attention import AttentionMetadata, TorchAttention, triton_available

pytestmark = pytest.mark.skipif(not triton_available(), reason="needs CUDA + triton")


def make_batch(query_lens, context_lens, num_heads, num_kv_heads, head_dim, block_size, dtype, seed=0):
    """Build a random paged KV cache with shuffled block assignment and matching metadata."""
    rng = random.Random(seed)
    torch.manual_seed(seed)
    dev = "cuda"
    seq_lens = [c + q for q, c in zip(query_lens, context_lens, strict=True)]
    blocks_per_seq = [-(-s // block_size) for s in seq_lens]
    num_blocks = sum(blocks_per_seq) + 8
    ids = list(range(num_blocks))
    rng.shuffle(ids)  # non-contiguous pages, like a real long-running pool
    tables, i = [], 0
    for n in blocks_per_seq:
        tables.append(ids[i : i + n])
        i += n
    max_blocks = max(blocks_per_seq)
    tables = [t + [0] * (max_blocks - len(t)) for t in tables]

    k_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, device=dev, dtype=dtype)
    v_cache = torch.randn_like(k_cache)
    q = torch.randn(sum(query_lens), num_heads, head_dim, device=dev, dtype=dtype)

    max_q = max(query_lens)
    q_gather, q_valid, starts, off = [], [], [0], 0
    for n in query_lens:
        q_gather.append(list(range(off, off + n)) + [0] * (max_q - n))
        q_valid.append([True] * n + [False] * (max_q - n))
        off += n
        starts.append(off)
    meta = AttentionMetadata(
        slot_mapping=torch.empty(0, dtype=torch.long, device=dev),
        block_tables=torch.tensor(tables, device=dev),
        seq_lens=torch.tensor(seq_lens, device=dev),
        query_lens=torch.tensor(query_lens, device=dev),
        q_gather=torch.tensor(q_gather, device=dev),
        q_valid=torch.tensor(q_valid, device=dev),
        query_start_loc=torch.tensor(starts, device=dev),
        max_seq_len=max(seq_lens),
        max_query_len=max_q,
    )
    return q, k_cache, v_cache, meta


CASES = {
    "decode": ([1] * 8, [5, 17, 64, 100, 1, 300, 16, 33]),
    "prefill": ([7, 64, 130], [0, 0, 0]),
    "chunked_prefill": ([50, 16, 3], [100, 16, 250]),
    "mixed": ([1, 1, 200, 1, 37], [40, 511, 0, 2, 70]),
    "long_decode": ([1, 1], [4000, 2047]),
}


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("heads", [(32, 8), (16, 16), (32, 4)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_matches_reference(case, heads, dtype):
    from kvserve.kernels.paged_attention import paged_attention

    query_lens, context_lens = CASES[case]
    num_heads, num_kv_heads = heads
    q, kc, vc, meta = make_batch(query_lens, context_lens, num_heads, num_kv_heads, 64, 16, dtype)
    scale = 64**-0.5
    ref = TorchAttention.forward(q.float(), kc.float(), vc.float(), meta, scale)
    out = paged_attention(q, kc, vc, meta.block_tables, meta.seq_lens, meta.query_start_loc, meta.max_query_len, scale)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


def test_head_dim_128_and_block_32():
    from kvserve.kernels.paged_attention import paged_attention

    q, kc, vc, meta = make_batch([1, 9, 1], [80, 40, 1000], 32, 8, 128, 32, torch.bfloat16)
    scale = 128**-0.5
    ref = TorchAttention.forward(q.float(), kc.float(), vc.float(), meta, scale)
    out = paged_attention(q, kc, vc, meta.block_tables, meta.seq_lens, meta.query_start_loc, meta.max_query_len, scale)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)
