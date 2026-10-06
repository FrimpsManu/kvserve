"""Load generator for OpenAI-compatible servers (kvserve, vLLM, SGLang, ...).

Sends streaming /v1/completions requests with Poisson arrivals (or all at once) and
reports TTFT, TPOT, inter-token latency, end-to-end latency and throughput, plus
goodput: the rate of requests meeting both latency SLOs.

    uv run python bench/bench_serving.py --base-url http://localhost:8000 \
        --num-prompts 200 --request-rate 8 --input-len 512 --output-len 128

Running the same command against vLLM on the same hardware gives a like-for-like
comparison, because prompts are fixed token-id lists and `ignore_eos` forces every
request to produce exactly `--output-len` tokens.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import numpy as np


@dataclass
class RequestResult:
    ok: bool
    prompt_len: int
    output_len: int = 0
    ttft: float = 0.0
    latency: float = 0.0
    itls: list[float] = field(default_factory=list)
    error: str = ""


def make_prompts(args: argparse.Namespace, vocab_size: int) -> list[tuple[list[int], int]]:
    """Random token prompts.

    `--shared-prefix-len` tokens at the start of each prompt are shared. With
    `--num-prefixes K`, there are K distinct shared prefixes (think K applications,
    each with its own long system prompt) and each request draws one, uniformly or
    Zipf-distributed so a few prefixes are hot. This is the workload where routing
    decides whether backends reuse each other's work or each recompute everything.
    """
    rng = random.Random(args.seed)

    def tokens(n: int) -> list[int]:
        return [rng.randrange(1000, vocab_size - 1000) for _ in range(n)]

    num_prefixes = max(1, args.num_prefixes)
    prefixes = [tokens(args.shared_prefix_len) for _ in range(num_prefixes)]
    if args.prefix_dist == "zipf":
        weights = [1.0 / (rank + 1) ** args.zipf_s for rank in range(num_prefixes)]
    else:
        weights = [1.0] * num_prefixes
    prompts = []
    for _ in range(args.num_prompts):
        n_in = max(1, int(args.input_len * rng.uniform(1 - args.range_ratio, 1 + args.range_ratio)))
        n_out = max(1, int(args.output_len * rng.uniform(1 - args.range_ratio, 1 + args.range_ratio)))
        prefix = rng.choices(prefixes, weights)[0]
        prompts.append((prefix + tokens(max(0, n_in - len(prefix))), n_out))
    return prompts


async def cache_counters(client: httpx.AsyncClient, urls: list[str]) -> tuple[int, int] | None:
    """Summed (prompt_tokens, cached_prompt_tokens) across kvserve backends' /stats."""
    total = cached = 0
    for url in urls:
        try:
            s = (await client.get(f"{url.rstrip('/')}/stats")).json()
        except (httpx.HTTPError, ValueError):
            return None
        total += s.get("prompt_tokens", 0)
        cached += s.get("cached_prompt_tokens", 0)
    return total, cached


async def send(client: httpx.AsyncClient, url: str, model: str, prompt: list[int], max_tokens: int) -> RequestResult:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    result = RequestResult(ok=False, prompt_len=len(prompt))
    start = last = time.perf_counter()
    try:
        async with client.stream("POST", url, json=payload) as resp:
            if resp.status_code != 200:
                result.error = f"HTTP {resp.status_code}: {(await resp.aread())[:200]!r}"
                return result
            async for line in resp.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                data = json.loads(line[6:])
                if data.get("usage"):
                    result.output_len = data["usage"]["completion_tokens"]
                if not data.get("choices"):
                    continue
                now = time.perf_counter()
                if result.ttft == 0.0:
                    result.ttft = now - start
                else:
                    result.itls.append(now - last)
                last = now
        result.latency = time.perf_counter() - start
        result.ok = True
    except Exception as e:  # noqa: BLE001 - record and keep the benchmark running
        result.error = repr(e)
    return result


async def run(args: argparse.Namespace) -> dict:
    base = args.base_url.rstrip("/")
    async with httpx.AsyncClient(timeout=httpx.Timeout(None, connect=10)) as client:
        model = args.model or (await client.get(f"{base}/v1/models")).json()["data"][0]["id"]
        prompts = make_prompts(args, args.vocab_size)
        url = f"{base}/v1/completions"
        limit = asyncio.Semaphore(args.max_concurrency or len(prompts))
        rng = np.random.default_rng(args.seed)

        async def limited(p: list[int], n: int) -> RequestResult:
            async with limit:
                return await send(client, url, model, p, n)

        # Warm up so model compilation / caches do not pollute the measurement.
        await send(client, url, model, prompts[0][0][:16], 4)
        paths_before = await step_paths(client, base)
        cache_before = await cache_counters(client, args.stats_urls) if args.stats_urls else None

        tasks = []
        start = time.perf_counter()
        for prompt, n_out in prompts:
            tasks.append(asyncio.create_task(limited(prompt, n_out)))
            if args.request_rate != float("inf"):
                await asyncio.sleep(rng.exponential(1.0 / args.request_rate))
        results = await asyncio.gather(*tasks)
        duration = time.perf_counter() - start
        paths_after = await step_paths(client, base)
        cache_after = await cache_counters(client, args.stats_urls) if args.stats_urls else None
    summary = summarize(args, model, results, duration)
    if cache_before is not None and cache_after is not None:
        prompt, cached = cache_after[0] - cache_before[0], cache_after[1] - cache_before[1]
        summary["prefix_cache"] = {"prompt_tokens": prompt, "cached_tokens": cached,
                                   "hit_rate": cached / prompt if prompt else 0.0}  # fmt: skip
    if paths_before is not None and paths_after is not None:
        summary["step_paths"] = {k: v - paths_before.get(k, 0) for k, v in paths_after.items()}
    return summary


async def step_paths(client: httpx.AsyncClient, base: str) -> dict[str, int] | None:
    """kvserve's per-execution-path step counters (None for servers without /stats)."""
    try:
        r = await client.get(f"{base}/stats")
        return r.json().get("step_paths") if r.status_code == 200 else None
    except (httpx.HTTPError, ValueError):
        return None


def _pct(values: list[float], *ps: int) -> dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values) * 1000  # ms
    stats = {"mean": float(arr.mean())}
    stats |= {f"p{p}": float(np.percentile(arr, p)) for p in ps}
    return stats


def summarize(args: argparse.Namespace, model: str, results: list[RequestResult], duration: float) -> dict:
    ok = [r for r in results if r.ok]
    tpots = [(r.latency - r.ttft) / (r.output_len - 1) for r in ok if r.output_len > 1]
    good = [
        r for r in ok
        if r.ttft * 1000 <= args.slo_ttft_ms
        and (r.output_len <= 1 or (r.latency - r.ttft) / (r.output_len - 1) * 1000 <= args.slo_tpot_ms)
    ]  # fmt: skip
    out_tokens = sum(r.output_len for r in ok)
    in_tokens = sum(r.prompt_len for r in ok)
    return {
        "model": model,
        "base_url": args.base_url,
        "label": args.label,
        "config": {k: v for k, v in vars(args).items() if k not in {"output"}},
        "host": platform.node(),
        "completed": len(ok),
        "failed": len(results) - len(ok),
        "errors": sorted({r.error for r in results if r.error})[:5],
        "duration_s": duration,
        "request_throughput": len(ok) / duration,
        "output_throughput": out_tokens / duration,
        "total_token_throughput": (in_tokens + out_tokens) / duration,
        "goodput": len(good) / duration,
        "ttft_ms": _pct([r.ttft for r in ok], 50, 90, 99),
        "tpot_ms": _pct(tpots, 50, 90, 99),
        "itl_ms": _pct([x for r in ok for x in r.itls], 50, 90, 99),
        "e2e_ms": _pct([r.latency for r in ok], 50, 90, 99),
    }


def report(s: dict) -> None:
    print(f"\n==== {s['label'] or s['model']} @ {s['base_url']} ====")
    print(f"completed {s['completed']}  failed {s['failed']}  duration {s['duration_s']:.1f}s")
    for e in s["errors"]:
        print("  error:", e)
    print(f"request throughput   {s['request_throughput']:9.2f} req/s")
    print(f"output throughput    {s['output_throughput']:9.1f} tok/s")
    print(f"total throughput     {s['total_token_throughput']:9.1f} tok/s")
    print(f"goodput (SLO met)    {s['goodput']:9.2f} req/s")
    for key in ("ttft_ms", "tpot_ms", "itl_ms", "e2e_ms"):
        st = s[key]
        if st:
            print(f"{key:<8} mean {st['mean']:9.1f}  p50 {st['p50']:9.1f}  p90 {st['p90']:9.1f}  p99 {st['p99']:9.1f}")
    if s.get("prefix_cache"):
        pc = s["prefix_cache"]
        tokens = f"{pc['cached_tokens']}/{pc['prompt_tokens']} prompt tokens"
        print(f"prefix cache hit     {100 * pc['hit_rate']:8.1f} %  ({tokens})")
    if s.get("step_paths"):
        total = sum(s["step_paths"].values()) or 1
        print("steps    " + "  ".join(f"{k} {v} ({100 * v / total:.0f}%)" for k, v in sorted(s["step_paths"].items())))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--model", default=None, help="defaults to the server's first model")
    p.add_argument("--label", default="")
    p.add_argument("--num-prompts", type=int, default=100)
    p.add_argument("--request-rate", type=float, default=float("inf"), help="req/s (Poisson); inf = all at once")
    p.add_argument("--max-concurrency", type=int, default=None)
    p.add_argument("--input-len", type=int, default=256)
    p.add_argument("--output-len", type=int, default=128)
    p.add_argument("--range-ratio", type=float, default=0.0, help="uniform +/- jitter on lengths")
    p.add_argument("--shared-prefix-len", type=int, default=0, help="common prefix tokens (prefix caching)")
    p.add_argument("--num-prefixes", type=int, default=1, help="distinct shared prefixes (e.g. system prompts)")
    p.add_argument("--prefix-dist", choices=["uniform", "zipf"], default="uniform", help="how requests pick a prefix")
    p.add_argument("--zipf-s", type=float, default=1.1, help="Zipf exponent for --prefix-dist zipf")
    p.add_argument("--stats-urls", type=lambda v: [u for u in v.split(",") if u], default=[],
                   help="kvserve backends to read prefix-cache counters from (e.g. behind a gateway)")  # fmt: skip
    p.add_argument("--vocab-size", type=int, default=128000)
    p.add_argument("--slo-ttft-ms", type=float, default=1000)
    p.add_argument("--slo-tpot-ms", type=float, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=None, help="append JSON result to this file")
    args = p.parse_args()

    summary = asyncio.run(run(args))
    report(summary)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("a") as f:
            f.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
