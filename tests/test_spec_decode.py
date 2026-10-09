"""Speculative decoding: n-gram proposals, draft scheduling, and output invariance."""

import numpy as np
import pytest

from kvserve import EngineConfig, LLMEngine, SamplingParams
from kvserve.kv_cache import KVCacheManager
from kvserve.scheduler import Scheduler
from kvserve.sequence import Sequence
from kvserve.spec_decode import max_draft_tokens, ngram_lookup


def test_ngram_lookup_proposes_continuation_of_latest_match():
    #               0  1  2  3  4  5  6  7  8  9
    tokens = np.array([5, 6, 7, 8, 1, 5, 6, 9, 2, 5, 6])
    # Suffix [5, 6] last occurred at 5..6, followed by 9, 2, 5.
    assert ngram_lookup(tokens, 3, n_max=3, n_min=2) == [9, 2, 5]
    # A longer suffix wins over a more recent shorter one.
    tokens = np.array([1, 2, 3, 4, 9, 2, 3, 7, 1, 2, 3])
    assert ngram_lookup(tokens, 2, n_max=3, n_min=2) == [4, 9]
    assert ngram_lookup(np.array([1, 2, 3, 4]), 4, n_max=3, n_min=2) == []  # no repeat
    assert ngram_lookup(np.array([4, 4]), 4, n_max=3, n_min=1) == [4]  # continuation ends at the suffix


def test_drafts_never_exceed_remaining_tokens():
    s = Sequence("a", [1, 2, 3], SamplingParams(max_tokens=3))
    assert max_draft_tokens(s, 4, max_model_len=100) == 2  # 3 tokens left: 2 drafts + 1 sampled
    s.output_token_ids = [0, 0]
    assert max_draft_tokens(s, 4, max_model_len=100) == 0
    assert max_draft_tokens(Sequence("b", [1] * 8, SamplingParams()), 4, max_model_len=10) == 1


def test_scheduler_adds_drafts_to_decodes_within_budget():
    kv = KVCacheManager(64, 4, enable_prefix_caching=False)
    sched = Scheduler(kv, max_num_seqs=8, max_num_batched_tokens=6)
    a, b = Sequence("a", [1, 2, 3], SamplingParams()), Sequence("b", [1, 2, 3], SamplingParams())
    for s in (a, b):
        sched.add(s)
    out = sched.schedule()  # prefill both
    for s, n in out.scheduled:
        s.num_computed_tokens += n
        s.append_token(9)
    a.spec_token_ids, b.spec_token_ids = [1, 2, 3, 4], [1, 2, 3, 4]
    out = sched.schedule()
    # a gets its pending token + 4 drafts (5 of 6), b only the pending token.
    assert [(s.request_id, n) for s, n in out.scheduled] == [("a", 5), ("b", 1)]
    assert a.spec_token_ids == [1, 2, 3, 4] and b.spec_token_ids == []
    assert len(a.block_table) == kv.num_blocks_for(4 + 4)


# ---- end to end (real model, CPU fp32) -----------------------------------------------

PROMPTS = [
    "Repeat this list exactly twice: alpha, beta, gamma, delta, epsilon, zeta.",
    "def fibonacci(n):",
    "The capital of France is",
    "Copy the sentence: The quick brown fox jumps over the lazy dog. The quick brown fox",
]
GREEDY = SamplingParams(temperature=0, max_tokens=40, ignore_eos=True)


def make_engine(**overrides) -> LLMEngine:
    return LLMEngine(EngineConfig(**{"device": "cpu", "num_kv_blocks": 256, **overrides}))


@pytest.fixture(scope="module")
def reference() -> list[list[int]]:
    return make_engine().generate(PROMPTS, GREEDY)


@pytest.mark.slow
@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"max_num_batched_tokens": 7},  # drafts trimmed to the budget, mixed with prefill chunks
        {"num_kv_blocks": 20, "block_size": 4},  # preemption: fits one sequence, not four
        {"num_speculative_tokens": 1},
    ],
    ids=["default", "tight-budget", "preemption", "k1"],
)
def test_ngram_greedy_is_output_invariant(reference, overrides):
    eng = make_engine(speculative_method="ngram", **overrides)
    assert eng.generate(PROMPTS, GREEDY) == reference
    assert eng.num_draft_tokens > 0 and eng.num_accepted_tokens > 0


@pytest.mark.slow
def test_ngram_with_prefix_caching_and_stop_tokens(reference):
    eng = make_engine(speculative_method="ngram", block_size=4)
    first = eng.generate(PROMPTS, GREEDY)
    second = eng.generate(PROMPTS, GREEDY)  # served largely from the prefix cache
    assert first == second == reference and eng.kv.prefix_hits > 0
    # Stopping mid-way through an accepted run of drafts truncates the output there.
    stop = reference[0][10]
    params = SamplingParams(temperature=0, max_tokens=40, stop_token_ids=(stop,))
    out = eng.generate(PROMPTS[:1], params)[0]
    assert out == reference[0][: reference[0].index(stop) + 1]


@pytest.mark.slow
def test_ngram_sampling_respects_max_tokens():
    params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=17, ignore_eos=True)
    outs = make_engine(speculative_method="ngram").generate(PROMPTS, params)
    assert all(len(o) == 17 for o in outs)
