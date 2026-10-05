"""Print a markdown comparison table from bench_serving.py JSONL results.

uv run python bench/summarize.py results/serving_*.jsonl
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict


def scenario(r: dict) -> str:
    c = r["config"]
    rate = c["request_rate"]
    rate_s = "burst" if rate in (None, float("inf")) or str(rate) in ("inf", "Infinity") else f"{rate:g} req/s"
    prefix = f", {c['shared_prefix_len']} shared" if c.get("shared_prefix_len") else ""
    return f"{c['input_len']}/{c['output_len']} tok, {rate_s}{prefix}"


def main() -> None:
    rows: dict[str, dict[str, dict]] = defaultdict(dict)
    for path in sys.argv[1:]:
        with open(path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    rows[scenario(r)][r["label"] or r["model"]] = r
    labels = sorted({label for by_label in rows.values() for label in by_label})
    print("| Scenario | System | Output tok/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | Goodput (req/s) |")
    print("|---|---|---|---|---|---|")
    for name, by_label in rows.items():
        for label in labels:
            r = by_label.get(label)
            if r is None:
                continue
            ttft, tpot = r["ttft_ms"], r["tpot_ms"]
            print(
                f"| {name} | {label} | {r['output_throughput']:.0f} | "
                f"{ttft.get('p50', 0):.0f} / {ttft.get('p99', 0):.0f} | "
                f"{tpot.get('p50', 0):.1f} / {tpot.get('p99', 0):.1f} | {r['goodput']:.2f} |"
            )


if __name__ == "__main__":
    main()
