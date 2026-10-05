"""End-to-end correctness against Hugging Face transformers (CPU, fp32, greedy).

Every scheduling feature must be output-invariant: chunked prefill, preemption and
prefix caching change *when* K/V are computed, never the generated tokens.
"""

import pytest
import torch

from kvserve import EngineConfig, LLMEngine, SamplingParams
from kvserve.attention import triton_available

pytestmark = pytest.mark.slow

PROMPTS = [
    "The capital of France is",
    "Explain paged attention in one sentence:",
    "def fibonacci(n):",
    "List three prime numbers greater than 100:",
]
GREEDY = SamplingParams(temperature=0, max_tokens=20, ignore_eos=True)


def make_engine(**overrides) -> LLMEngine:
    return LLMEngine(EngineConfig(**{"device": "cpu", "num_kv_blocks": 256, **overrides}))


@pytest.fixture(scope="module")
def engine() -> LLMEngine:
    return make_engine()


@pytest.fixture(scope="module")
def reference(engine) -> list[list[int]]:
    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(engine.runner.model_path, dtype=torch.float32).eval()
    outs = []
    for p in PROMPTS:
        ids = engine.tokenizer(p, return_tensors="pt").input_ids
        gen = hf.generate(ids, max_new_tokens=20, min_new_tokens=20, do_sample=False)
        outs.append(gen[0, ids.shape[1] :].tolist())
    return outs


def test_matches_hf(engine, reference):
    assert engine.generate(PROMPTS, GREEDY) == reference


def test_single_vs_batched(engine, reference):
    assert [engine.generate([p], GREEDY)[0] for p in PROMPTS] == reference


def test_chunked_prefill(reference):
    eng = make_engine(max_num_batched_tokens=3)  # prompts split into many chunks
    assert eng.generate(PROMPTS, GREEDY) == reference


def test_preemption(reference):
    eng = make_engine(num_kv_blocks=10, block_size=4)  # fits one sequence, far too small for 4 sequences at once
    preempted = 0
    ids = [eng.add_request(p, GREEDY) for p in PROMPTS]
    results = {}
    while eng.has_unfinished():
        for out in eng.step():
            if out.finished:
                results[out.request_id] = out.output_token_ids
        preempted += eng.last_step.num_preempted if eng.last_step else 0
    assert preempted > 0
    assert [results[i] for i in ids] == reference


def test_prefix_caching_hits_and_is_exact(reference):
    eng = make_engine(block_size=4)
    system = "You are a helpful assistant. Answer concisely and accurately. "
    prompts = [system + p for p in PROMPTS]
    first = eng.generate(prompts[:1], GREEDY)
    rest = eng.generate(prompts[1:], GREEDY)
    assert eng.kv.prefix_hits > 0

    cold = make_engine(block_size=4, enable_prefix_caching=False)
    assert first + rest == cold.generate(prompts, GREEDY)


@pytest.mark.skipif(not triton_available(), reason="needs CUDA + triton")
def test_triton_backend_matches_hf(reference):
    eng = LLMEngine(EngineConfig(device="cuda", dtype="float32", attention_backend="triton", num_kv_blocks=256))
    assert eng.generate(PROMPTS, GREEDY) == reference
    # Mixed prefill/decode steps and chunked prefill through the kernel.
    chunked = LLMEngine(
        EngineConfig(device="cuda", dtype="float32", attention_backend="triton", max_num_batched_tokens=5)
    )
    assert chunked.generate(PROMPTS, GREEDY) == reference


def test_sampling_is_seeded_and_respects_max_tokens():
    params = SamplingParams(temperature=0.8, top_p=0.9, top_k=50, max_tokens=12, ignore_eos=True)
    a = make_engine(seed=7).generate(PROMPTS[:2], params)
    b = make_engine(seed=7).generate(PROMPTS[:2], params)
    assert a == b and all(len(o) == 12 for o in a)
