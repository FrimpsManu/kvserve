# kvserve

An LLM inference engine built from first principles: paged KV cache, continuous
batching with chunked prefill, prefix caching, recompute preemption, a Triton
paged-attention kernel, and an OpenAI-compatible streaming server with Prometheus
metrics. Benchmarked head-to-head against vLLM on the same GPU.

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
  prefix cache, LRU eviction           (Triton kernel on CUDA; torch reference)
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
`--no-enable-prefix-caching`, `--attention-backend auto|torch|triton`, `--device`, `--dtype`.

## Benchmarks

`bench/bench_serving.py` drives any OpenAI-compatible server (kvserve, vLLM, SGLang)
with fixed token-id prompts and `ignore_eos`, so results are directly comparable.
It reports TTFT, TPOT, inter-token latency, end-to-end latency, throughput and goodput
under latency SLOs.

```bash
uv run python bench/bench_serving.py --base-url http://localhost:8000 \
  --num-prompts 64 --input-len 256 --output-len 64
```

All GPU results: RTX 4090 (24 GB), Llama-3.2-1B-Instruct, bf16, block size 16. Raw data
in [`results/rtx4090/`](results/rtx4090).

### Paged attention kernel (decode)

`bench/bench_kernel.py`, 32 query / 8 KV heads, head_dim 64, L2 flushed before every
timed call, median of 50. Decode attention is memory bound, so achieved HBM bandwidth
against the 1008 GB/s peak is the figure of merit.

| Batch | Context | Triton | Torch reference | Speedup | Bandwidth | % of peak |
|---|---|---|---|---|---|---|
| 1 | 2048 | 0.108 ms | 0.224 ms | 2.1x | 39 GB/s | 4% |
| 8 | 2048 | 0.115 ms | 0.549 ms | 4.8x | 293 GB/s | 29% |
| 32 | 8192 | 0.649 ms | 6.670 ms | 10.3x | 827 GB/s | 82% |
| 64 | 8192 | 1.193 ms | 13.150 ms | 11.0x | **900 GB/s** | **89%** |
| 128 | 2048 | 0.634 ms | 6.859 ms | 10.8x | 847 GB/s | 84% |

Large batches run near the bandwidth roofline. Small batches are underutilised: the
grid is (sequences x KV heads), so batch 1 launches only 8 programs on a 128-SM GPU.
Split-KV (flash-decoding) is the planned fix.

### Serving: kvserve vs. vLLM 0.30

`bench/compare_vllm.sh`: both servers on the same pod, same model, `max_num_seqs=128`,
`max_num_batched_tokens=2048`, 8 GB KV cache, prefix caching on, 256 requests per
scenario, fixed token-id prompts with `ignore_eos`. Goodput counts requests meeting
TTFT <= 1 s and TPOT <= 100 ms.

| Scenario (in/out tokens) | System | Output tok/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | Goodput (req/s) |
|---|---|---|---|---|---|
| 512/128, burst | kvserve | 2656 | 4669 / 9254 | 29.0 / 32.8 | 5.03 |
| | vLLM | 6946 | 2272 / 4051 | 9.3 / 12.8 | 21.20 |
| 512/128, 4 req/s | kvserve | 447 | 28 / 46 | 12.5 / 14.0 | 3.49 |
| | vLLM | 453 | 22 / 35 | 3.4 / 3.7 | 3.54 |
| 512/128, 8 req/s | kvserve | 870 | 32 / 63 | 14.2 / 16.9 | 6.80 |
| | vLLM | 901 | 18 / 32 | 3.4 / 3.6 | 7.04 |
| 512/128, 16 req/s | kvserve | 1620 | 48 / 82 | 19.6 / 21.8 | 12.66 |
| | vLLM | 1773 | 18 / 26 | 3.5 / 3.7 | 13.85 |
| 512/128, 32 req/s | kvserve | 2622 | 120 / 1063 | 26.2 / 40.0 | 19.69 |
| | vLLM | 3423 | 17 / 35 | 3.8 / 4.0 | 26.75 |
| 1024/128, 768 shared prefix, 16 req/s | kvserve | 1619 | 50 / 197 | 21.2 / 24.3 | 12.65 |
| | vLLM | 1764 | 22 / 40 | 3.8 / 4.1 | 13.79 |

**Where kvserve stands:** at 4-16 req/s it delivers 91-99% of vLLM's goodput; at
saturation it reaches 38% of vLLM's peak throughput, and per-token latency is 3.7-7x
higher.

**Why (measured, not guessed):** a per-step breakdown shows the model forward pass
takes ~10 ms whether the batch holds 1 or 128 sequences, against ~2.5 ms of actual
GPU work for a 1B model. The step is bound by CPU-side kernel launch overhead
(~300 small launches through Python per step), not by the GPU or the attention
kernel. Consistent with that, the same code reached 1700 tok/s on a host with a
slower CPU and 2656 tok/s on this one, with the same GPU. CUDA Graphs for decode is the
next milestone; vLLM's 3.4 ms TPOT is the target.

### Apple M5 Pro (MPS), development baseline

Torch reference attention, 64 concurrent requests:

| Workload | Prompt len | TTFT mean | TPOT mean | Output tok/s |
|---|---|---|---|---|
| Unique prompts | 256 | 1456 ms | 85 ms | 587 |
| 448-token shared prefix | 512 | **575 ms** | 118 ms | 511 |

Prefix caching cuts TTFT by 60% despite 2x longer prompts. Offline decode throughput
scales from 45 tok/s at batch 1 to 999 tok/s at batch 32 (22x) from continuous
batching alone.

### Reproducing on a GPU pod

```bash
scripts/sync_to_pod.sh root@<ip> <port>            # from your laptop
bash scripts/pod_setup.sh --vllm                   # on the pod
uv run pytest                                      # 56 tests incl. GPU kernel + e2e
uv run python bench/bench_kernel.py --peak-gbps 1008
bash bench/compare_vllm.sh
```

The pod needs an NVIDIA driver new enough for current vLLM wheels (CUDA 13 hosts);
kvserve itself pins CUDA 12.8 wheels and runs on driver 570+.

## Testing

```bash
uv run pytest -m "not slow"   # allocator, scheduler, invariants (seconds)
uv run pytest                 # + token-exact comparison against HF transformers
```

The slow suite asserts identical greedy output to `transformers` for: batched vs. single
requests, chunked prefill with a 3-token budget, heavy preemption with a 10-block pool,
and prefix-cache hits vs. a cold cache. On CUDA, the same comparison runs end to end
through the Triton kernel (fp32, IEEE dots). The kernel is also tested against the
torch reference on shuffled page layouts across decode, prefill, chunked prefill, mixed
batches, long contexts, GQA ratios, dtypes and head sizes. A randomized test checks
that the allocator never leaks or double-frees blocks.

## Roadmap

- [x] Paged KV cache, continuous batching, chunked prefill, preemption
- [x] Prefix caching (chained block hashes, LRU eviction)
- [x] OpenAI-compatible streaming server, Prometheus metrics, benchmark harness
- [x] Triton paged-attention kernel (mixed prefill/decode, GQA-aware), 89% of HBM peak
- [x] NVIDIA benchmarks vs. vLLM, with per-step profiling of the gap
- [ ] CUDA Graphs for decode (target: vLLM-level TPOT)
- [ ] Split-KV decode for small batches / long contexts
- [ ] Tensor parallelism (NCCL)
- [ ] Go gateway, KV-cache-aware router, disaggregated prefill/decode
- [ ] Speculative decoding, FP8 KV cache
