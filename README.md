# kvserve

An LLM inference engine built from first principles: paged KV cache, continuous
batching with chunked prefill, prefix caching, recompute preemption, and an
OpenAI-compatible streaming server with Prometheus metrics.

Every scheduling optimisation is verified to be **output-invariant**: generations are
token-for-token identical to Hugging Face `transformers` under batching, chunked
prefill, preemption and prefix caching.

## Architecture

```
 HTTP clients ──► FastAPI server (OpenAI API, SSE streaming, /metrics)
                        │  asyncio queues
                        ▼
                 AsyncLLMEngine ── dedicated engine thread (no locks on engine state)
                        │
                        ▼
                    LLMEngine: schedule ─► execute ─► sample ─► update
                     │                 │
          ┌──────────┘                 └──────────┐
          ▼                                       ▼
      Scheduler                              ModelRunner
  token budget per step,              flat (unpadded) token batch,
  decode-first, chunked prefill,      attention metadata, sampling
  recompute preemption                          │
          │                                     ▼
          ▼                              Llama model (fused QKV / gate-up)
   KVCacheManager                               │
  block allocator, block tables,                ▼
  ref counts, chained-hash             Paged attention backend
  prefix cache, LRU eviction           (torch reference; Triton/CUDA next)
```

### Key design decisions

| Decision | Why |
|---|---|
| One invariant, `num_computed_tokens < num_tokens`, drives scheduling | Prefill, chunked prefill, decode and post-preemption recompute are the same code path; no separate prefill phase |
| Per-step token budget, running sequences served first | Bounds step latency (protects TPOT of in-flight decodes) while new prompts fill leftover capacity |
| Fixed-size KV blocks + block tables | No contiguous allocation per sequence; waste is at most one partial block per sequence |
| Chained SHA-256 block hashes for prefix caching | A block hit implies the whole prefix matches; freed blocks stay cached until LRU eviction |
| Recompute (not swap) preemption | Simple and cheap with prefix caching; no CPU swap space to manage |
| Engine on its own thread, commands via queue | GPU work never blocks the event loop; engine state is single-threaded |
| Fused QKV and gate/up projections | Fewer, larger GEMMs: better hardware utilisation |

## Quickstart

```bash
uv sync
uv run kvserve serve --port 8000            # downloads Llama-3.2-1B-Instruct on first run
```

```bash
curl localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is a KV cache?"}],"max_tokens":64,"stream":true}'
```

Python:

```python
from kvserve import LLMEngine, EngineConfig, SamplingParams

engine = LLMEngine(EngineConfig())
print(engine.generate(["The capital of France is"], SamplingParams(temperature=0, max_tokens=16)))
```

Engine flags (`uv run kvserve serve --help`): `--block-size`, `--kv-cache-memory-gb`,
`--max-num-seqs`, `--max-num-batched-tokens`, `--max-model-len`,
`--no-enable-prefix-caching`, `--device`, `--dtype`.

## Benchmarks

`bench/bench_serving.py` drives any OpenAI-compatible server (kvserve, vLLM, SGLang)
with fixed token-id prompts and `ignore_eos`, so results are directly comparable.
It reports TTFT, TPOT, inter-token latency, end-to-end latency, throughput and goodput
under latency SLOs.

```bash
uv run python bench/bench_serving.py --base-url http://localhost:8000 \
  --num-prompts 64 --input-len 256 --output-len 64
```

Preliminary results, Apple M5 Pro (MPS, bf16), Llama-3.2-1B-Instruct, 64 concurrent requests,
torch reference attention:

| Workload | Prompt len | TTFT mean | TPOT mean | Output tok/s |
|---|---|---|---|---|
| Unique prompts | 256 | 1456 ms | 85 ms | 587 |
| 448-token shared prefix | 512 | **575 ms** | 118 ms | 511 |

Offline decode throughput (128 output tokens each) scales from 45 tok/s at batch 1 to
999 tok/s at batch 32 (22x), from continuous batching alone.

NVIDIA GPU results and a head-to-head with vLLM are in progress.

## Testing

```bash
uv run pytest -m "not slow"   # allocator, scheduler, invariants (seconds)
uv run pytest                 # + token-exact comparison against HF transformers
```

The slow suite asserts identical greedy output to `transformers` for: batched vs. single
requests, chunked prefill with a 3-token budget, heavy preemption with a 10-block pool,
and prefix-cache hits vs. a cold cache. A randomized test checks that the allocator
never leaks or double-frees blocks.

## Roadmap

- [x] Paged KV cache, continuous batching, chunked prefill, preemption
- [x] Prefix caching (chained block hashes, LRU eviction)
- [x] OpenAI-compatible streaming server, Prometheus metrics, benchmark harness
- [ ] Triton paged-attention kernel (decode + prefill), CUDA Graphs for decode
- [ ] NVIDIA benchmarks vs. vLLM
- [ ] Tensor parallelism (NCCL)
- [ ] Go gateway, KV-cache-aware router, disaggregated prefill/decode
- [ ] Speculative decoding, FP8 KV cache
