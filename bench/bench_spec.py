"""Speculative decoding benchmark on real text: no speculation vs n-gram vs a draft model.

Random-token prompts (bench_serving.py) say nothing about speculation, whose speedup
depends entirely on how predictable the output is. Two workloads with chat-formatted
prompts:

- chat:     open-ended questions; the output is new text, so only a draft model helps.
- grounded: code edits, summaries and extraction over a given document; the output
            copies long spans of the input, which n-gram lookup predicts for free.

For each method and batch size, requests run to completion in waves of `batch`
(greedy, natural EOS, up to --max-tokens). Reports output throughput, the speedup over
no speculation, the acceptance rate, mean tokens emitted per verification, and how
many outputs are token-identical to the no-speculation run.

    uv run python bench/bench_spec.py --model unsloth/Llama-3.1-8B-Instruct \
        --draft-model unsloth/Llama-3.2-1B-Instruct --batch-sizes 1 8 32 \
        --output results/spec_rtx4090/spec.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from kvserve import EngineConfig, LLMEngine, SamplingParams

ROOT = Path(__file__).resolve().parents[1]

CHAT = [
    "Explain how a hash map handles collisions, with a short example.",
    "Write a short story (about 150 words) about a lighthouse keeper who finds a message in a bottle.",
    "What are the main differences between TCP and UDP? Give one use case for each.",
    "Give me a 5-day beginner workout plan with rest days.",
    "Why is the sky blue? Explain it to a 10-year-old.",
    "Compare Python lists and tuples, and say when to use each.",
    "Draft a polite email asking a professor for a recommendation letter.",
    "What causes inflation, and how do central banks respond to it?",
    "Describe the water cycle in four steps.",
    "Write a haiku sequence (three haiku) about autumn in a city.",
    "How does public-key cryptography let two strangers communicate securely?",
    "Suggest a weekend itinerary for a first visit to Chicago.",
    "What is gradient descent? Include the update rule.",
    "Summarize the plot of Romeo and Juliet in one paragraph.",
    "Give three tips for writing clean, maintainable code, with a reason for each.",
    "Explain what a database index is and the trade-offs of adding one.",
]


def grounded_prompts() -> list[str]:
    def src(path: str, start: int = 0, lines: int = 60) -> str:
        return "".join((ROOT / path).read_text().splitlines(keepends=True)[start : start + lines])

    readme = (ROOT / "README.md").read_text()
    readme_intro = readme[: readme.index("## Demo")]
    return [
        "Add a one-line comment above every function in this Python code. Return the complete updated code "
        "and nothing else.\n\n```python\n" + src("src/kvserve/sampler.py", 0, 50) + "```",
        "Rename the variable `seq` to `sequence` everywhere in this code. Return the complete updated code.\n\n"
        "```python\n" + src("src/kvserve/scheduler.py", 40, 70) + "```",
        "Add type hints where they are missing and return the full code.\n\n```python\n"
        + src("src/kvserve/kv_cache.py", 100, 60)
        + "```",
        "Convert this Python code to use `pathlib` instead of string paths where it touches files, and return "
        "the complete updated file.\n\n```python\n" + src("src/kvserve/config.py", 0, 60) + "```",
        "Summarize this document in about 8 bullet points, quoting the exact numbers it reports.\n\n" + readme_intro,
        "List every design decision in this document with its one-sentence justification, as a numbered list, "
        "keeping the original wording.\n\n" + readme_intro,
        "Fix any typos in this text and return the full corrected text.\n\n" + readme_intro[:3000],
        "Rewrite this function's docstring in Google style and return the whole function.\n\n```python\n"
        + src("src/kvserve/sampler.py", 50, 60)
        + "```",
    ]


def chat_ids(eng: LLMEngine, prompt: str) -> list[int]:
    text = eng.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                             add_generation_prompt=True)  # fmt: skip
    return eng.tokenizer.encode(text, add_special_tokens=False)


def build(args: argparse.Namespace, method: str) -> LLMEngine:
    cfg = EngineConfig(
        model=args.model,
        kv_cache_memory_gb=args.kv_gb,
        max_num_seqs=max(args.batch_sizes),
        max_model_len=args.max_model_len,
        speculative_method=method,
        num_speculative_tokens=args.k_draft if method == "draft" else args.k_ngram,
        draft_model=args.draft_model if method == "draft" else "",
    )
    return LLMEngine(cfg)


def run_waves(eng: LLMEngine, prompts: list[list[int]], batch: int, params: SamplingParams) -> dict:
    eng.num_draft_tokens = eng.num_accepted_tokens = eng.num_verify_steps = 0
    outputs: list[list[int]] = []
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    sync()
    start = time.perf_counter()
    for i in range(0, len(prompts), batch):
        ids = [eng.add_request(p, params) for p in prompts[i : i + batch]]
        done: dict[str, list[int]] = {}
        while eng.has_unfinished():
            for out in eng.step():
                if out.finished:
                    done[out.request_id] = out.output_token_ids
        outputs += [done[r] for r in ids]
    sync()
    elapsed = time.perf_counter() - start
    tokens = sum(len(o) for o in outputs)
    return {
        "outputs": outputs,
        "elapsed_s": elapsed,
        "output_tokens": tokens,
        "tok_per_s": tokens / elapsed,
        "draft_tokens": eng.num_draft_tokens,
        "accepted_tokens": eng.num_accepted_tokens,
        "acceptance": eng.num_accepted_tokens / eng.num_draft_tokens if eng.num_draft_tokens else None,
        # Tokens a verifying sequence emits per target pass: accepted drafts + 1.
        "tokens_per_verify": 1 + eng.num_accepted_tokens / eng.num_verify_steps if eng.num_verify_steps else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="unsloth/Llama-3.1-8B-Instruct")
    ap.add_argument("--draft-model", default="unsloth/Llama-3.2-1B-Instruct")
    ap.add_argument("--methods", nargs="+", default=["none", "ngram", "draft"])
    ap.add_argument("--workloads", nargs="+", default=["chat", "grounded"])
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--k-ngram", type=int, default=4)
    ap.add_argument("--k-draft", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--kv-gb", type=float, default=3.0)
    ap.add_argument("--repeat", type=int, default=1, help="cycle each workload's prompts this many times")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    params = SamplingParams(temperature=0, max_tokens=args.max_tokens)
    gpu = torch.cuda.get_device_name() if torch.cuda.is_available() else "none"
    rows: list[dict] = []
    baseline: dict[tuple[str, int], dict] = {}
    for method in args.methods:
        eng = build(args, method)
        chat = [chat_ids(eng, p) for p in CHAT]
        grounded = [chat_ids(eng, p) for p in grounded_prompts()]
        workloads = {"chat": chat, "grounded": grounded}
        run_waves(eng, chat[:2], 2, SamplingParams(temperature=0, max_tokens=16))  # warm-up
        for name in args.workloads:
            prompts = workloads[name] * args.repeat
            for batch in args.batch_sizes:
                # Every batch size sees whole waves: cycle prompts up to a multiple of `batch`.
                n = max(len(prompts), batch)
                n += -n % batch
                wave_prompts = [prompts[i % len(prompts)] for i in range(n)]
                r = run_waves(eng, wave_prompts, batch, params)
                base = baseline.setdefault((name, batch), r) if method == "none" else baseline.get((name, batch))
                same = sum(a == b for a, b in zip(r["outputs"], base["outputs"], strict=True)) if base else None
                row = {
                    "gpu": gpu, "model": args.model, "method": method, "workload": name, "batch": batch,
                    "k": {"none": 0, "ngram": args.k_ngram, "draft": args.k_draft}[method],
                    "draft_model": args.draft_model if method == "draft" else None, "requests": n,
                    **{key: v for key, v in r.items() if key != "outputs"},
                    "speedup": r["tok_per_s"] / base["tok_per_s"] if base else None,
                    "identical_outputs": f"{same}/{n}" if same is not None else None,
                }  # fmt: skip
                rows.append(row)
                acc = f"{row['acceptance']:.0%}" if row["acceptance"] is not None else "-"
                tpv = f"{row['tokens_per_verify']:.2f}" if row["tokens_per_verify"] else "-"
                spd = f"{row['speedup']:.2f}x" if row["speedup"] else "-"
                print(
                    f"{method:6} {name:9} batch {batch:3}  {row['tok_per_s']:8.1f} tok/s  speedup {spd:6} "
                    f"accept {acc:4}  tok/verify {tpv:5}  identical {row['identical_outputs']}",
                    flush=True,
                )
        del eng
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
