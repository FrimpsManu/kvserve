from kvserve.kv_cache import KVCacheManager
from kvserve.scheduler import Scheduler
from kvserve.sequence import SamplingParams, Sequence, SequenceStatus


def make_scheduler(num_blocks=64, block_size=4, max_seqs=8, budget=32):
    kv = KVCacheManager(num_blocks, block_size, enable_prefix_caching=False)
    return Scheduler(kv, max_seqs, budget), kv


def seq(rid: str, n: int) -> Sequence:
    return Sequence(rid, list(range(n)), SamplingParams())


def run_step(sched: Scheduler):
    """Apply a schedule as the engine would, emitting one token per completed sequence."""
    out = sched.schedule()
    for s, n in out.scheduled:
        s.num_computed_tokens += n
        if s.num_computed_tokens == s.num_tokens:
            s.append_token(0)
    return out


def test_token_budget_and_chunked_prefill():
    sched, _ = make_scheduler(budget=16)
    long = seq("long", 40)
    sched.add(long)
    chunks = [run_step(sched).num_tokens for _ in range(3)]
    assert chunks == [16, 16, 8]
    assert long.output_token_ids == [0]  # sampled only after the final chunk


def test_running_decodes_are_served_before_new_prefills():
    sched, _ = make_scheduler(budget=16)
    a = seq("a", 10)
    sched.add(a)
    run_step(sched)  # a fully prefilled
    b = seq("b", 40)
    sched.add(b)
    out = run_step(sched)
    assert out.scheduled[0] == (a, 1)  # decode first
    assert out.scheduled[1] == (b, 15)  # prefill gets the rest of the budget


def test_max_num_seqs():
    sched, _ = make_scheduler(max_seqs=2, budget=100)
    for i in range(4):
        sched.add(seq(str(i), 4))
    out = run_step(sched)
    assert len(out.scheduled) == 2 and len(sched.waiting) == 2


def test_preemption_when_kv_pool_is_full():
    sched, kv = make_scheduler(num_blocks=4, block_size=4, budget=100)
    a, b = seq("a", 8), seq("b", 7)
    sched.add(a)
    sched.add(b)
    run_step(sched)  # a: 2 blocks, b: 2 blocks, pool full
    assert kv.num_free_blocks == 0
    out = run_step(sched)  # a needs a 3rd block -> b (newest) is preempted
    assert out.num_preempted == 1
    assert b.status is SequenceStatus.WAITING and b.num_computed_tokens == 0
    assert sched.waiting[0] is b
    assert [s for s, _ in out.scheduled] == [a]


def test_finish_releases_blocks():
    sched, kv = make_scheduler()
    a = seq("a", 10)
    sched.add(a)
    run_step(sched)
    sched.finish(a)
    assert kv.num_free_blocks == kv.num_blocks and not sched.has_unfinished
