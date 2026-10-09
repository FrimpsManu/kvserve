# kvserve

An LLM inference engine built from first principles: paged KV cache, continuous
batching with chunked prefill, prefix caching, recompute preemption, a Triton
paged-attention kernel, decode and piecewise CUDA graphs, and an OpenAI-compatible
streaming server whose engine runs in its own process. Benchmarked head-to-head against
vLLM on the same GPU: **98-100% of vLLM's goodput up to 32 req/s and 87% of its burst
throughput** on an RTX 4090 (Ryzen 9 7950X host). Speculative decoding (n-gram lookup
or a draft model) makes Llama-3.1-8B **1.65x faster on chat and up to 2.5x faster on
grounded tasks** at batch 1.

Every scheduling optimisation is verified to be **output-invariant**: generations are
token-for-token identical to Hugging Face `transformers` under batching, chunked
prefill, preemption and prefix caching.

## Architecture

```
 HTTP clients
      │
      ▼
 ┌───────────────────────── API server process ─────────────────────────┐
 │ FastAPI: OpenAI API, SSE streaming, /metrics, /stats, demo page        │
 │ ProcessEngineClient: tokenize, O(1) incremental detokenize, telemetry  │
 └───────────────▲─────────────────────────────────────┬─────────────────┘
                 │ one message per step:               │ add / abort
                 │ all requests' new tokens + stats    │ (ZeroMQ IPC)
 ┌───────────────┴──────────── engine process ─────────▼─────────────────┐
 │ LLMEngine: schedule ─► execute ─► sample ─► update                     │
 │                                                                        │
 │  Scheduler                         ModelRunner                         │
 │  token budget per step,            decode CUDA graphs │ piecewise      │
 │  decode-first, chunked prefill,    graphs │ eager, flat token batch    │
 │  recompute preemption                         │                        │
 │      │                             Llama model (fused QKV / gate-up),  │
 │  KVCacheManager                    split into pieces around attention  │
 │  block tables, ref counts,                    │                        │
 │  chained-hash prefix cache,        Triton paged attention (torch ref.  │
 │  LRU eviction                      on CPU/MPS)                         │
 └────────────────────────────────────────────────────────────────────────┘
```

### Key design decisions

| Decision | Why |
|---|---|
| One invariant, `num_computed_tokens < num_tokens`, drives scheduling | Prefill, chunked prefill, decode and post-preemption recompute are the same code path; no separate prefill phase |
| Per-step token budget, running sequences served first | Bounds step latency (protects TPOT of in-flight decodes) while new prompts fill leftover capacity |
| Fixed-size KV blocks + block tables | No contiguous allocation per sequence; waste is at most one partial block per sequence |
| Chained SHA-256 block hashes for prefix caching | A block hit implies the whole prefix matches; freed blocks stay cached until LRU eviction |
| Recompute (not swap) preemption | Simple and cheap with prefix caching; no CPU swap space to manage |
| Engine in its own process, ZeroMQ IPC, one message per step | API server and engine no longer share a GIL; IPC cost scales with steps, not tokens x requests (`--engine-mode thread` keeps the single-process design) |
| CUDA graphs: full graphs for decode, piecewise graphs around attention for mixed steps | Decode was launch bound (~300 kernel launches per step); piecewise graphs extend that to steps mixing prefill and decode |
| Fused QKV and gate/up projections | Fewer, larger GEMMs: better hardware utilisation |
| Speculative drafts ride the chunked-prefill path; the draft model shares the target's block tables | Verification is just a multi-token step, and rejected K/V sits past `num_computed_tokens`, so nothing rolls back; one allocator manages both models' caches |

## Demo

`uv run kvserve serve`, then open **http://localhost:8000**: a chat UI plus a live view of
the engine.

![kvserve demo dashboard: chat on the left, live throughput, scheduler and KV cache panels on the right](docs/demo.png)

- **Per-reply stats:** time to first token, decode tok/s, and how many prompt tokens
  came from the prefix cache. The second turn of a conversation typically reuses most of
  the prompt (e.g. 48 of 77 prompt tokens on the second turn).
- **Live engine panel** (polls `/stats`): throughput over the last 60 s, running and
  waiting sequences, tokens in the last forward pass, KV cache usage, prefix-cache hit rate.
- **Continuous batching demo:** fire 8/32/64 concurrent requests and watch them share one
  batch; reports aggregate throughput, per-request decode speed and TTFT.

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
`--no-enable-prefix-caching`, `--attention-backend auto|torch|triton`, `--no-enable-cuda-graphs`,
`--no-enable-piecewise-graphs`, `--max-piecewise-tokens`, `--device`, `--dtype`.

## Benchmarks

`bench/bench_serving.py` drives any OpenAI-compatible server (kvserve, vLLM, SGLang)
with fixed token-id prompts and `ignore_eos`, so results are directly comparable.
It reports TTFT, TPOT, inter-token latency, end-to-end latency, throughput and goodput
under latency SLOs.

```bash
uv run python bench/bench_serving.py --base-url http://localhost:8000 \
  --num-prompts 64 --input-len 256 --output-len 64
```

All GPU results: RTX 4090 (24 GB), Llama-3.2-1B-Instruct (speculative decoding: Llama-3.1-8B), bf16, block size 16. Raw data
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

### CUDA Graphs for decode

Decode was launch bound: ~300 small kernels per step issued from Python took ~10 ms
for ~2.5 ms of GPU work. Uniform decode steps now replay a CUDA graph captured per
padded batch size (`bench/bench_decode.py`, 512-token contexts):

| Batch | Eager | CUDA graph | Speedup | Decode tok/s |
|---|---|---|---|---|
| 1 | 11.90 ms | **3.95 ms** | 3.0x | 253 |
| 8 | 12.20 ms | 4.64 ms | 2.6x | 1,725 |
| 32 | 12.88 ms | 5.45 ms | 2.4x | 5,872 |
| 128 | 14.08 ms | 8.81 ms | 1.6x | **14,526** |

The remaining growth with batch size is per-step Python input preparation, which
scales with the number of sequences.

### Piecewise CUDA graphs for mixed steps

Decode graphs only cover steps where every sequence decodes one token. Under load most
steps mix prefill chunks with decodes, so they still paid full launch overhead. The
forward pass is split at the attention calls into `num_layers + 1` pieces; each piece is
captured per token-count bucket and attention runs eagerly in between with the step's
own metadata. Pieces never touch the KV cache, so padding is harmless.

Single-step time, one sequence prefilling (RTX 4090):

| Tokens in step | Eager | Piecewise | |
|---|---|---|---|
| 17 | 11.6 ms | **3.6 ms** | 3.2x |
| 257 | 11.4 ms | **6.4 ms** | 1.8x |
| 513 | 11.7 ms | 10.1 ms | 1.2x |
| 1025 | 18.4 ms | 20.5 ms | 0.9x (padded to 1280) |
| 2048 | 33.9 ms | 33.9 ms | 1.0x |

Past ~512 tokens a step is compute bound, so piecewise graphs cover steps up to
`--max-piecewise-tokens` (default 512) and larger steps run eagerly. Capture: 408
graphs, about 1 s at startup.

### Serving: engine process vs. thread vs. vLLM 0.31

The latest results, all on one pod: RTX 4090 with an AMD Ryzen 9 7950X host. kvserve
is still partly CPU bound, so this fast desktop CPU flatters both kvserve modes; the
process-vs-thread comparison is the like-for-like number. Same settings as below.

| Scenario (in/out tokens) | System | Output tok/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | Goodput (req/s) |
|---|---|---|---|---|---|
| 512/128, burst | kvserve | **6639** | 2221 / 4133 | 11.3 / 12.8 | 20.26 |
| | kvserve thread | 5475 | 2576 / 4949 | 14.8 / 16.5 | 15.20 |
| | vLLM | 7673 | 1953 / 3645 | 8.5 / 11.0 | 23.42 |
| 512/128, 4 req/s | kvserve | 453 | 15 / 29 | 4.5 / 4.9 | 3.54 |
| | kvserve thread | 453 | 16 / 28 | 4.8 / 5.3 | 3.54 |
| | vLLM | 453 | 15 / 23 | 3.4 / 3.6 | 3.54 |
| 512/128, 16 req/s | kvserve | 1759 | 10 / 16 | 4.8 / 5.0 | 13.75 |
| | kvserve thread | 1755 | 11 / 18 | 5.3 / 5.7 | 13.71 |
| | vLLM | 1775 | 12 / 15 | 3.5 / 3.7 | 13.87 |
| 512/128, 32 req/s | kvserve | 3376 | 11 / 18 | 5.1 / 5.3 | **26.38** |
| | kvserve thread | 3353 | 14 / 23 | 6.1 / 6.7 | 26.20 |
| | vLLM | 3432 | 13 / 18 | 3.7 / 3.9 | 26.82 |
| 1024/128, 768 shared prefix, 16 req/s | kvserve | 1751 | 14 / 22 | 5.3 / 5.6 | 13.68 |
| | kvserve thread | 1746 | 14 / 24 | 5.9 / 6.3 | 13.64 |
| | vLLM | 1767 | 15 / 20 | 3.7 / 4.0 | 13.80 |

(8 req/s in the raw data: equal goodput, TPOT 4.6 vs 4.9 vs 3.4 ms.)

- **Burst: +21% from the engine process** (6,639 vs 5,475 tok/s), reaching 87% of vLLM.
- **Stable across restarts:** three fresh servers per mode measured 6,608 / 6,608 / 6,623
  tok/s (process) and 5,521 / 5,466 / 5,509 (thread), versus ~35% swings between restarts
  in the single-process design on an EPYC host
  ([`burst_restarts.jsonl`](results/rtx4090_ryzen7950x/burst_restarts.jsonl)).
- **Up to 32 req/s:** 98-100% of vLLM's goodput with equal TTFT; TPOT is 1.1-1.6 ms
  higher than vLLM.

### Serving ablation: CUDA graphs (EPYC 7642 host)

`bench/compare_vllm.sh`: all systems on one pod (RTX 4090, EPYC 7642), same model,
`max_num_seqs=128`, `max_num_batched_tokens=2048`, 8 GB KV cache, prefix caching on,
256 requests per scenario, fixed token-id prompts with `ignore_eos`. Goodput counts
requests meeting TTFT <= 1 s and TPOT <= 100 ms. Ablation: **kvserve** (decode +
piecewise graphs), **decode graphs** (`--no-enable-piecewise-graphs`), **eager**
(`--no-enable-cuda-graphs`).

| Scenario (in/out tokens) | System | Output tok/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | Goodput (req/s) |
|---|---|---|---|---|---|
| 512/128, 4 req/s | kvserve | 452 | 20 / 35 | 5.5 / 6.7 | 3.53 |
| | decode graphs | 452 | 21 / 37 | 5.3 / 6.4 | 3.53 |
| | eager | 446 | 30 / 53 | 13.9 / 15.9 | 3.49 |
| | vLLM | 453 | 24 / 39 | 3.5 / 3.7 | 3.54 |
| 512/128, 8 req/s | kvserve | 895 | 13 / 61 | 5.6 / 7.9 | 6.99 |
| | decode graphs | 893 | 24 / 43 | 6.6 / 9.4 | 6.98 |
| | eager | 866 | 36 / 85 | 15.5 / 25.0 | 6.77 |
| | vLLM | 901 | 14 / 22 | 3.4 / 3.6 | 7.04 |
| 512/128, 16 req/s | kvserve | 1730 | 22 / 42 | **8.4** / 12.1 | 13.52 |
| | decode graphs | 1721 | 46 / 90 | 12.1 / 17.5 | 13.45 |
| | eager | 1625 | 45 / 105 | 18.2 / 29.9 | 12.70 |
| | vLLM | 1770 | 20 / 33 | 3.6 / 4.0 | 13.83 |
| 512/128, 32 req/s | kvserve | 2634 | 70 / **523** | 27.0 / 31.7 | **20.58** |
| | decode graphs | 2309 | 427 / 2910 | 37.1 / 43.8 | 9.72 |
| | eager | 2072 | 91 / 1900 | 37.7 / 45.8 | 12.78 |
| | vLLM | 3400 | 25 / 49 | 4.0 / 4.4 | 26.57 |
| 1024/128, 768 shared prefix, 16 req/s | kvserve | 1725 | 31 / 77 | **10.5** / 16.0 | 13.48 |
| | decode graphs | 1716 | 49 / 140 | 14.9 / 20.9 | 13.40 |
| | eager | 1490 | 84 / 167 | 38.1 / 45.5 | 11.64 |
| | vLLM | 1760 | 29 / 56 | 4.1 / 4.6 | 13.75 |

**Where kvserve stands:** 98-100% of vLLM's goodput at 4-16 req/s and **77% at 32 req/s**
(up from 37% with decode graphs alone), with p99 TTFT at 32 req/s down from 2.9 s to
0.5 s.

**Burst (all 256 requests at once) is a serving-layer problem, not an engine one**
(fixed since by running the engine in its own process, see above).
Offline, the engine finishes the same burst at 6,730-6,980 tok/s, on par with vLLM's
served ~6,700. Over HTTP it reaches 2,500-3,600 tok/s, varying by up to ~35% across
server restarts (run-to-run variance within one server instance is ~2%; raw data in
[`burst_variance.jsonl`](results/rtx4090/burst_variance.jsonl)). Engine metrics during an
HTTP burst show why: the engine is busy 98% of the time but needs 60% more steps (466 vs
290: requests trickle in through the HTTP layer, so batches are smaller) and each step is
15% slower. The API server and the engine share one Python process and its GIL, so
per-token streaming work (detokenisation, JSON events for 256 streams) competes with the
engine thread. vLLM runs the engine in a separate process for this reason; that is the
next milestone.

Earlier results on other hosts are kept in [`results/rtx4090/`](results/rtx4090): decode
graphs vs eager vs vLLM 0.30 (EPYC 7532) and the pre-graphs baseline (EPYC 75F3). For a
launch-bound engine the host CPU changes results, so only same-host numbers are compared.

### Speculative decoding (Llama-3.1-8B, RTX 4090)

A decoding sequence proposes k draft tokens; the target scores its pending token and
all drafts in one forward pass (the same multi-token path as chunked prefill) and
rejection sampling keeps the longest agreeing prefix plus one token of its own, so a
step emits 1 to k + 1 tokens. Rejection sampling keeps the target's output
distribution exactly; greedy decoding stays token-for-token identical.

Two proposers: **n-gram lookup** (`--speculative-method ngram`) matches the last 2-4
tokens earlier in the context and proposes what followed: free, and strong when the
output copies its input. A **draft model** (`--speculative-method draft --draft-model
unsloth/Llama-3.2-1B-Instruct`) runs a small model with its own K/V pool indexed by the
target's block tables, so the target's allocator, preemption and prefix caching cover
it too; its k - 1 single-token passes replay CUDA graphs.

`bench/bench_spec.py`: Llama-3.1-8B-Instruct target, bf16, greedy, up to 256 output
tokens, k = 4, real chat-formatted prompts. *chat* is 16 open-ended questions;
*grounded* is 8 code edits, summaries and extractions over a given document. Speedup is
end-to-end output throughput (prefill included); per-request decode is tokens after the
first over the time after the first, the speed a streaming user sees.

| Workload | Method | Batch 1 | Batch 8 | Batch 32 | Acceptance | Tokens per verify |
|---|---|---|---|---|---|---|
| chat | none | 57 tok/s | 399 tok/s | 1,427 tok/s | | |
| chat | n-gram | 1.08x | 0.98x | 0.91x | 21% | 1.8 |
| chat | draft 1B | **1.65x** | 1.21x (decode 1.45x) | 1.15x (decode 1.37x) | 65% | 3.6 |
| grounded | none | 53 tok/s | 400 tok/s | 1,424 tok/s | | |
| grounded | n-gram | **2.21x** (decode 2.55x) | 1.32x (decode 2.32x) | 1.19x (decode 1.98x) | 61-64% | 3.6 |
| grounded | draft 1B | 1.80x | 1.29x (decode 1.72x) | 1.23x (decode 1.51x) | 82-85% | 4.3 |

- **Batch 1 is where speculation pays**: an 8B decode step is memory bound (18.5 ms),
  and verifying 5 tokens costs about the same as decoding 1 (20.4 ms measured).
- **The gain shrinks with batch size** as steps become compute bound and verification
  work stops being free. End-to-end throughput falls faster than per-request decode
  speed because requests run in waves that last as long as their longest member.
- **N-gram needs copyable output.** On open-ended chat it rarely matches (21%
  acceptance) and costs up to 10% at batch 32; on grounded tasks it beats the draft
  model at no cost at all. The draft model helps both.

Tokens per step (k), batch 1, end-to-end speedup:

| k | n-gram, grounded | draft, chat | draft, grounded |
|---|---|---|---|
| 2 | 1.86x | 1.37x | 1.45x |
| 4 | 2.21x | 1.65x | 1.80x |
| 6 | **2.47x** (decode 3.11x) | **1.73x** | **1.93x** |

Larger k keeps paying at batch 1; at batch 8 k = 4 and k = 6 are about equal (1.32x / 1.37x
n-gram grounded, 1.21x / 1.20x draft chat), so the default stays k = 4.

**Exactness.** In fp32 on the GPU, every output of every method at every batch size is
token-identical to decoding without speculation (192 / 192, Llama-3.2-1B). In bf16
some long greedy outputs diverge (e.g. 4 / 16 chat outputs identical at batch 1):
scoring k + 1 tokens in one pass rounds differently from scoring one, which can flip a
near-tie, after which the continuations differ. Raw data in
[`results/spec_rtx4090/`](results/spec_rtx4090).

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
uv run pytest                                      # 61 tests incl. GPU kernel, CUDA graphs, e2e
uv run python bench/bench_decode.py                # eager vs CUDA graph decode step
uv run python bench/bench_kernel.py --peak-gbps 1008
bash bench/compare_vllm.sh
uv run python bench/bench_spec.py --kv-gb 2         # speculative decoding, 8B target + 1B draft
```

The pod needs an NVIDIA driver new enough for current vLLM wheels (CUDA 13 hosts);
kvserve itself pins CUDA 12.8 wheels and runs on driver 570+.

## Distributed serving: kvgateway

`gateway/` is a Go reverse proxy that routes OpenAI-compatible requests across several
kvserve instances. Each instance has its own prefix cache, so *where* a request runs
decides whether its shared prefix (a long system prompt, a conversation so far) is
reused or recomputed.

```bash
cd gateway && go build -o bin/kvgateway ./cmd/kvgateway
bin/kvgateway --backends http://gpu0:8000,http://gpu1:8000 --policy prefix_affinity
```

| Policy | How it picks a backend |
|---|---|
| `round_robin` | In turn. Baseline: every instance ends up computing and caching every prefix. |
| `least_loaded` | Fewest requests in flight from the gateway (instant and exact for one gateway). |
| `prefix_affinity` | Rendezvous hashing on the prompt's first 512 characters (or 128 token ids), with **bounded loads**: an instance is eligible only below `ceil((1 + eps) * average)` in-flight, so a hot prefix spills to its second choice instead of overloading one instance. |

It streams SSE through without buffering, fails over only on connection errors (never
after a backend has started generating), health-checks and restores backends, propagates
client disconnects, and exports Prometheus metrics.

### Does cache-aware routing beat load-based routing?

`bench/compare_routing.sh`: 4 kvserve instances on 2x RTX 4090 (2 per GPU, 1 GB KV cache
each), EPYC 7763 host. 32 shared prefixes of 1,792 tokens with Zipf popularity (a few hot
"applications"), ~2,048-token prompts, 128 output tokens. Together the prefixes exceed one
instance's cache, so routing determines reuse. Instances restart with cold caches for
every run; means over 3 seeds (2 at 64 req/s), ranges in brackets.

| Rate | Policy | Prefix-cache hit | Prompt tokens recomputed | Output tok/s | TTFT p50 / p99 (ms) | TPOT p50 (ms) |
|---|---|---|---|---|---|---|
| 16 req/s | round robin | 59.6% [58.5-61.5] | 40.4% | 1,975 | 46 / 144 | 13.5 |
| | least loaded | 58.8% [57.1-62.2] | 41.2% | 1,975 | 48 / 159 | 13.4 |
| | **prefix affinity** | **74.0%** [73.6-74.7] | **26.0%** | 1,973 | 44 / 134 | **12.9** |
| 32 req/s | round robin | 59.9% [58.9-61.8] | 40.1% | 3,653 | 52 / 252 | 16.2 |
| | least loaded | 60.7% [60.1-61.4] | 39.3% | 3,663 | 55 / 211 | 16.0 |
| | **prefix affinity** | **74.9%** [73.6-76.9] | **25.1%** | 3,670 | 48 / 180 | **14.7** |
| 64 req/s (saturated) | round robin | 63.0% [62.1-63.9] | 37.0% | 4,714 | 2,485 / 3,915 | 18.0 |
| | least loaded | 62.8% [62.5-63.0] | 37.2% | 4,695 | 2,527 / 3,672 | 18.3 |
| | **prefix affinity** | **77.7%** [75.8-79.6] | **22.3%** | **5,374** | **1,337 / 1,875** | **16.4** |

- **Prefix affinity recomputes 36-40% fewer prompt tokens** than either baseline at every
  rate, in all 8 rate and seed combinations.
- **Below saturation** the fleet has spare capacity, so throughput is unchanged; the saved
  prefill shows up as 4-9% lower TPOT (less prefill competing with decode) and a slightly
  lower median TTFT. Tail TTFT at these rates is mixed across seeds, so no tail claim.
- **At saturation** the saved work becomes capacity: **+14% throughput** (+10% to +18% by
  seed) and median TTFT **2,485 -> 1,337 ms**.
- Load-only routing (least loaded) is no better than round robin for cache reuse: it
  balances queues but scatters prefixes.

Raw data: [`results/routing_2x4090/`](results/routing_2x4090).

## Testing

```bash
uv run pytest -m "not slow"   # allocator, scheduler, invariants (seconds)
uv run pytest                 # + token-exact comparison against HF transformers
```

The slow suite asserts identical greedy output to `transformers` for: batched vs. single
requests, chunked prefill with a 3-token budget, heavy preemption with a 10-block pool,
and prefix-cache hits vs. a cold cache. Speculative decoding is checked the same way:
greedy output with n-gram or draft-model speculation equals output without it under a
tight token budget, preemption, prefix caching and stop tokens, using the target as
its own draft (near-total acceptance) and a tiny random-weight draft (mostly rejected).
At every step the draft model's first proposal must match running it from scratch,
which catches stale draft K/V after full acceptance, rejection or preemption, and the
rejection sampler is checked statistically to reproduce the target distribution. On CUDA, the same comparison runs end to end
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
- [x] CUDA Graphs for decode: 3.95 ms/step at batch 1 (3.0x), 96-100% of vLLM goodput at moderate load
- [x] Piecewise CUDA graphs for mixed prefill/decode steps: 77% of vLLM goodput at 32 req/s (from 37%)
- [x] Engine in its own process + O(1) incremental detokenisation: +21% burst throughput, 87% of vLLM
- [ ] Split-KV decode for small batches / long contexts
- [ ] Tensor parallelism (NCCL)
- [x] Go gateway with prefix-affinity routing: 36-40% less prompt recompute, +14% throughput at saturation vs round robin
- [ ] Disaggregated prefill/decode
- [x] Speculative decoding (n-gram and draft model): 1.65x on chat, up to 2.5x on grounded tasks at batch 1 (Llama-3.1-8B)
- [ ] FP8 KV cache
