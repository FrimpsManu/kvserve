"""CUDA graph decode path (CUDA only)."""

import pytest
import torch

from kvserve import EngineConfig, LLMEngine, SamplingParams
from kvserve.attention import triton_available
from kvserve.cuda_graph import capture_sizes, piecewise_sizes

pytestmark = pytest.mark.skipif(not triton_available(), reason="needs CUDA + triton")

PROMPTS = [
    "The capital of France is",
    "Explain paged attention in one sentence:",
    "def fibonacci(n):",
    "List three prime numbers greater than 100:",
    "Write a haiku about GPUs:",
]


def test_capture_sizes():
    assert capture_sizes(1) == [1]
    assert capture_sizes(8) == [1, 2, 4, 8]
    assert capture_sizes(40) == [1, 2, 4, 8, 16, 32, 40]
    assert capture_sizes(128)[-3:] == [96, 112, 128]


def test_piecewise_sizes():
    sizes = piecewise_sizes(2048)
    assert sizes[:5] == [1, 2, 4, 8, 16] and sizes[-1] == 2048
    assert sizes == sorted(set(sizes))
    assert piecewise_sizes(100)[-1] == 100 and piecewise_sizes(3)[-1] == 3


def make(graphs: bool, **kw) -> LLMEngine:
    cfg = {"device": "cuda", "attention_backend": "triton", "num_kv_blocks": 512, "enable_cuda_graphs": graphs}
    return LLMEngine(EngineConfig(**(cfg | kw)))


@pytest.mark.slow
def test_graph_matches_eager_fp32():
    """Same tokens with and without graphs, including padded buckets (5 -> 8) and a
    shrinking batch as sequences finish at different lengths."""
    params = [SamplingParams(temperature=0, max_tokens=8 + 4 * i, ignore_eos=True) for i in range(len(PROMPTS))]

    def run(eng: LLMEngine) -> list[list[int]]:
        ids = [eng.add_request(p, sp) for p, sp in zip(PROMPTS, params, strict=True)]
        out = {}
        while eng.has_unfinished():
            for o in eng.step():
                if o.finished:
                    out[o.request_id] = o.output_token_ids
        return [out[i] for i in ids]

    eager = run(make(False, dtype="float32"))
    graphed_engine = make(True, dtype="float32")
    assert graphed_engine.runner.graphs is not None
    assert run(graphed_engine) == eager


@pytest.mark.slow
def test_graph_replay_matches_eager_hidden_states():
    eng = make(True, dtype="float32")
    runner = eng.runner
    for p in PROMPTS:
        eng.add_request(p, SamplingParams(temperature=0, max_tokens=3, ignore_eos=True))
    eng.step()  # prefill everything; next step is a uniform decode
    sched = eng.scheduler.schedule()
    assert all(n == 1 for _, n in sched.scheduled)

    seqs = [s for s, _ in sched.scheduled]
    ids = [s.token_ids[-1] for s in seqs]
    pos = [s.num_computed_tokens for s in seqs]
    slots = [eng.kv.slot(s, s.num_computed_tokens) for s in seqs]
    lens = [s.num_computed_tokens + 1 for s in seqs]
    graph_hidden = runner.graphs.run(ids, pos, slots, [s.block_table for s in seqs], lens).clone()

    meta = runner._build_metadata(sched, slots, lens, [1] * len(seqs))
    eager_hidden = runner.model(
        torch.tensor(ids, device="cuda"), torch.tensor(pos, device="cuda"), runner.kv_caches, meta
    )
    torch.testing.assert_close(graph_hidden, eager_hidden, atol=1e-4, rtol=1e-4)


@pytest.mark.slow
def test_graph_padding_never_touches_allocated_blocks():
    eng = make(True)
    scratch = eng.runner.scratch_block
    assert scratch == eng.kv.num_blocks  # outside the allocator's pool
    for p in PROMPTS[:3]:  # batch of 3 replays the 4-bucket with one padding row
        eng.add_request(p, SamplingParams(temperature=0, max_tokens=6, ignore_eos=True))
    while eng.has_unfinished():
        eng.step()
        assert all(b != scratch for s in eng.scheduler.running for b in s.block_table)


@pytest.mark.slow
def test_matches_hf_with_graphs():
    from transformers import AutoModelForCausalLM

    eng = make(True, dtype="float32")
    greedy = SamplingParams(temperature=0, max_tokens=20, ignore_eos=True)
    ours = eng.generate(PROMPTS[:4], greedy)
    hf = AutoModelForCausalLM.from_pretrained(eng.runner.model_path, dtype=torch.float32).eval()
    for prompt, out in zip(PROMPTS[:4], ours, strict=True):
        ids = eng.tokenizer(prompt, return_tensors="pt").input_ids
        ref = hf.generate(ids, max_new_tokens=20, min_new_tokens=20, do_sample=False)[0, ids.shape[1] :]
        assert out == ref.tolist()


@pytest.mark.slow
def test_piecewise_matches_eager_on_mixed_steps():
    """A 24-token budget forces chunked prefill steps that mix with decodes; every step
    must go through graphs and produce the same tokens as eager execution."""
    params = SamplingParams(temperature=0, max_tokens=12, ignore_eos=True)

    def run(eng: LLMEngine) -> list[list[int]]:
        ids = []
        for p in PROMPTS:  # staggered arrivals: new prompts join a running batch
            ids.append(eng.add_request(p * 3, params))
            eng.step()
        out = {}
        while eng.has_unfinished():
            for o in eng.step():
                if o.finished:
                    out[o.request_id] = o.output_token_ids
        return [out[i] for i in ids]

    eager = run(make(False, dtype="float32", max_num_batched_tokens=24))
    eng = make(True, dtype="float32", max_num_batched_tokens=24)
    assert eng.runner.piecewise.max_tokens == 24  # capped by the step budget
    assert run(eng) == eager
    paths = eng.runner.step_paths
    assert paths["piecewise"] > 0 and paths["decode_graph"] > 0 and paths["eager"] == 0, paths


@pytest.mark.slow
def test_piecewise_matches_hf():
    from transformers import AutoModelForCausalLM

    eng = make(True, dtype="float32", max_num_batched_tokens=7, max_graph_batch_size=1)
    greedy = SamplingParams(temperature=0, max_tokens=10, ignore_eos=True)
    ours = eng.generate(PROMPTS[:3], greedy)  # batch of 3 > decode graph max -> piecewise
    assert eng.runner.step_paths["piecewise"] > 0
    hf = AutoModelForCausalLM.from_pretrained(eng.runner.model_path, dtype=torch.float32).eval()
    for prompt, out in zip(PROMPTS[:3], ours, strict=True):
        ids = eng.tokenizer(prompt, return_tensors="pt").input_ids
        ref = hf.generate(ids, max_new_tokens=10, min_new_tokens=10, do_sample=False)[0, ids.shape[1] :]
        assert out == ref.tolist()


@pytest.mark.slow
def test_large_steps_bypass_piecewise():
    """Steps above max_piecewise_tokens are compute bound and run eagerly."""
    eng = make(True, max_num_batched_tokens=256, max_piecewise_tokens=64)
    assert eng.runner.piecewise.max_tokens == 64
    eng.generate([[1000 + i for i in range(200)]], SamplingParams(temperature=0, max_tokens=2, ignore_eos=True))
    assert eng.runner.step_paths["eager"] >= 1  # the 200-token prefill
