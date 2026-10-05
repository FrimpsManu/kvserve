import random

import pytest

from kvserve.kv_cache import KVCacheManager
from kvserve.sequence import SamplingParams, Sequence


def make_seq(tokens: list[int], rid: str = "r") -> Sequence:
    return Sequence(rid, tokens, SamplingParams())


def compute(kv: KVCacheManager, seq: Sequence, n: int) -> None:
    """Simulate the engine computing n tokens of seq."""
    assert kv.allocate_slots(seq, n)
    seq.num_computed_tokens += n
    kv.cache_full_blocks(seq)


def test_allocate_and_free_returns_all_blocks():
    kv = KVCacheManager(num_blocks=8, block_size=4)
    seq = make_seq(list(range(10)))
    assert kv.allocate_slots(seq, 10)
    assert len(seq.block_table) == 3  # ceil(10 / 4)
    assert kv.num_free_blocks == 5
    kv.free(seq)
    assert kv.num_free_blocks == 8 and seq.block_table == []


def test_decode_allocates_new_block_only_at_boundary():
    kv = KVCacheManager(num_blocks=8, block_size=4, enable_prefix_caching=False)
    seq = make_seq([1, 2, 3])
    compute(kv, seq, 3)
    seq.append_token(4)
    compute(kv, seq, 1)  # fills block 0
    assert len(seq.block_table) == 1
    seq.append_token(5)
    compute(kv, seq, 1)
    assert len(seq.block_table) == 2


def test_allocation_failure_changes_nothing():
    kv = KVCacheManager(num_blocks=2, block_size=4)
    seq = make_seq(list(range(12)))
    assert not kv.allocate_slots(seq, 12)
    assert seq.block_table == [] and kv.num_free_blocks == 2


def test_slot_mapping():
    kv = KVCacheManager(num_blocks=8, block_size=4)
    seq = make_seq(list(range(6)))
    kv.allocate_slots(seq, 6)
    b0, b1 = seq.block_table
    assert [kv.slot(seq, p) for p in range(6)] == [b0 * 4, b0 * 4 + 1, b0 * 4 + 2, b0 * 4 + 3, b1 * 4, b1 * 4 + 1]


@pytest.mark.parametrize(("start", "n"), [(0, 1), (0, 4), (3, 1), (3, 6), (2, 13), (8, 4), (5, 0)])
def test_slots_matches_per_token_slot(start, n):
    kv = KVCacheManager(num_blocks=16, block_size=4)
    seq = make_seq(list(range(20)))
    kv.allocate_slots(seq, 20)
    seq.block_table.reverse()  # non-contiguous, non-monotonic pages
    assert kv.slots(seq, start, n) == [kv.slot(seq, p) for p in range(start, start + n)]


def test_prefix_cache_hit_shares_blocks():
    kv = KVCacheManager(num_blocks=16, block_size=4)
    a = make_seq(list(range(10)), "a")
    compute(kv, a, 10)
    b = make_seq(list(range(8)) + [99, 98, 97], "b")
    hits = kv.find_cached_prefix(b)
    assert hits == a.block_table[:2]  # two full shared blocks
    assert kv.allocate_slots(b, b.num_tokens - 8, hits)
    assert b.num_computed_tokens == 8
    assert kv.ref_counts[hits[0]] == 2


def test_prefix_cache_always_leaves_one_token_to_compute():
    kv = KVCacheManager(num_blocks=16, block_size=4)
    a = make_seq(list(range(8)), "a")
    compute(kv, a, 8)
    b = make_seq(list(range(8)), "b")  # identical prompt, exactly two blocks
    assert len(kv.find_cached_prefix(b)) == 1


def test_freed_cached_block_is_revived_then_evicted_lru():
    kv = KVCacheManager(num_blocks=4, block_size=4)
    a = make_seq(list(range(8)), "a")
    compute(kv, a, 8)
    kv.free(a)
    assert kv.num_free_blocks == 4
    b = make_seq(list(range(8)) + [7], "b")
    hits = kv.find_cached_prefix(b)
    assert len(hits) == 2  # still cached after free
    assert kv.allocate_slots(b, 1, hits)
    assert kv.num_free_blocks == 1
    kv.free(b)
    # Filling the pool with unrelated data evicts the cached blocks.
    c = make_seq(list(range(100, 116)), "c")
    compute(kv, c, 16)
    assert kv.find_cached_prefix(make_seq(list(range(9)), "d")) == []


def test_different_prefix_same_block_contents_does_not_hit():
    kv = KVCacheManager(num_blocks=16, block_size=4)
    a = make_seq([1, 2, 3, 4, 5, 6, 7, 8, 0], "a")
    compute(kv, a, 9)
    b = make_seq([9, 9, 9, 9, 5, 6, 7, 8, 0], "b")  # block 1 equal, but chained hash differs
    assert kv.find_cached_prefix(b) == []


@pytest.mark.parametrize("seed", range(5))
def test_random_workload_never_leaks(seed):
    rng = random.Random(seed)
    kv = KVCacheManager(num_blocks=32, block_size=4)
    live: list[Sequence] = []
    for step in range(300):
        if live and rng.random() < 0.3:
            kv.free(live.pop(rng.randrange(len(live))))
            continue
        prefix = [rng.randrange(3)] * rng.randrange(0, 12)
        seq = make_seq(prefix + [rng.randrange(50) for _ in range(rng.randrange(1, 10))], f"s{step}")
        hits = kv.find_cached_prefix(seq)
        n = seq.num_tokens - len(hits) * 4
        if kv.allocate_slots(seq, n, hits):
            seq.num_computed_tokens += n
            kv.cache_full_blocks(seq)
            live.append(seq)
        in_use = {b for s in live for b in s.block_table}
        assert all(kv.ref_counts[b] > 0 for b in in_use)
        assert len(in_use) + kv.num_free_blocks == kv.num_blocks
    for s in live:
        kv.free(s)
    assert kv.num_free_blocks == kv.num_blocks and all(r == 0 for r in kv.ref_counts)
