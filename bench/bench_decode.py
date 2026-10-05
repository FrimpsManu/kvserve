"""Offline decode-step benchmark: eager vs CUDA graphs.

Prefills `batch` prompts, then times pure decode steps (one token per sequence),
reporting ms/step and output tokens/s. Isolates the engine from HTTP and scheduling
noise; this is the number CUDA graphs are meant to move.

    uv run python bench/bench_decode.py --batch-sizes 1 8 32 128
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from kvserve import EngineConfig, LLMEngine, SamplingParams


def measure(eng: LLMEngine, batch: int, prompt_len: int, steps: int) -> float:
    params = SamplingParams(temperature=0, max_tokens=steps + 10, ignore_eos=True)
    for i in range(batch):
        eng.add_request([1000 + (i * 7 + j) % 30000 for j in range(prompt_len)], params)
    while eng.scheduler.waiting or any(s.num_computed_tokens < s.num_tokens - 1 for s in eng.scheduler.running):
        eng.step()  # finish prefill
    for _ in range(3):
        eng.step()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        eng.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    for seq in list(eng.scheduler.running):
        eng.abort(seq.request_id)
    return elapsed / steps * 1000


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32, 64, 128])
    p.add_argument("--prompt-len", type=int, default=512)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    rows = []
    results: dict[bool, dict[int, float]] = {}
    for graphs in (False, True):
        eng = LLMEngine(EngineConfig(device="cuda", kv_cache_memory_gb=8, max_num_seqs=128, enable_cuda_graphs=graphs))
        results[graphs] = {b: measure(eng, b, args.prompt_len, args.steps) for b in args.batch_sizes}
        del eng
        torch.cuda.empty_cache()

    gpu = torch.cuda.get_device_name()
    print(f"{gpu}, prompt {args.prompt_len} tokens, {args.steps} decode steps")
    print(f"{'batch':>5} {'eager ms/step':>14} {'graph ms/step':>14} {'speedup':>8} {'graph tok/s':>12}")
    for b in args.batch_sizes:
        e, g = results[False][b], results[True][b]
        print(f"{b:>5} {e:>14.2f} {g:>14.2f} {e / g:>7.2f}x {b / g * 1000:>12.0f}")
        rows.append({"gpu": gpu, "batch": b, "eager_ms": e, "graph_ms": g, "graph_tok_s": b / g * 1000})
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


if __name__ == "__main__":
    main()
