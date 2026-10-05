"""Decode attention microbenchmark: Triton kernel vs. torch reference vs. hardware peak.

Decode attention is memory bound: each step must read every cached K/V byte once.
Achieved bandwidth = KV bytes read / kernel time, compared with the GPU's peak HBM
bandwidth, is the honest figure of merit (FLOPs are irrelevant here).

    uv run python bench/bench_kernel.py --peak-gbps 300   # L4 = 300, RTX 4090 = 1008, H100 SXM = 3350
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from test_triton_attention import make_batch  # noqa: E402

from kvserve.attention import TorchAttention  # noqa: E402
from kvserve.kernels.paged_attention import paged_attention  # noqa: E402


def time_ms(fn, iters: int = 50) -> float:
    for _ in range(5):
        fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--peak-gbps", type=float, default=None, help="GPU peak HBM bandwidth in GB/s")
    p.add_argument("--num-heads", type=int, default=32)
    p.add_argument("--num-kv-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    dtype = torch.bfloat16
    gpu = torch.cuda.get_device_name()
    print(f"{gpu}  heads={args.num_heads}/{args.num_kv_heads}  head_dim={args.head_dim}  bf16  block_size=16")
    print(f"{'batch':>5} {'ctx':>6} {'triton ms':>10} {'torch ms':>9} {'speedup':>8} {'GB/s':>8} {'% peak':>7}")
    rows = []
    for batch in (1, 8, 32, 64, 128):
        for ctx in (512, 2048, 8192):
            if batch * ctx > 64 * 8192:
                continue
            q, kc, vc, meta = make_batch(
                [1] * batch, [ctx - 1] * batch, args.num_heads, args.num_kv_heads, args.head_dim, 16, dtype
            )
            scale = args.head_dim**-0.5

            def run_triton(q=q, kc=kc, vc=vc, meta=meta, scale=scale):
                return paged_attention(q, kc, vc, meta.block_tables, meta.seq_lens, meta.query_start_loc, 1, scale)

            def run_torch(q=q, kc=kc, vc=vc, meta=meta, scale=scale):
                return TorchAttention.forward(q, kc, vc, meta, scale)

            t_triton, t_torch = time_ms(run_triton), time_ms(run_torch)
            kv_bytes = 2 * batch * ctx * args.num_kv_heads * args.head_dim * 2
            gbps = kv_bytes / (t_triton / 1000) / 1e9
            pct = f"{100 * gbps / args.peak_gbps:6.1f}%" if args.peak_gbps else "    -"
            speedup = t_torch / t_triton
            print(f"{batch:>5} {ctx:>6} {t_triton:>10.3f} {t_torch:>9.3f} {speedup:>7.1f}x {gbps:>8.0f} {pct}")
            rows.append({"gpu": gpu, "batch": batch, "ctx": ctx, "triton_ms": t_triton, "torch_ms": t_torch,
                         "gbps": gbps, "peak_gbps": args.peak_gbps})  # fmt: skip
            del q, kc, vc, meta
            torch.cuda.empty_cache()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


if __name__ == "__main__":
    main()
