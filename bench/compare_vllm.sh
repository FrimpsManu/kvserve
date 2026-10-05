#!/usr/bin/env bash
# Like-for-like serving benchmark: kvserve vs. vLLM, same GPU, model, scheduler limits and prompts.
#   bash bench/compare_vllm.sh            (run on the GPU pod after scripts/pod_setup.sh --vllm)
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/workspace/hf}
MODEL=${MODEL:-unsloth/Llama-3.2-1B-Instruct}
KV_GB=${KV_GB:-8}
MAX_SEQS=${MAX_SEQS:-128}
MAX_TOKENS=${MAX_TOKENS:-2048}
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')
OUT=results/serving_${GPU}_$(date +%Y%m%d_%H%M%S).jsonl
mkdir -p results

wait_healthy() { for _ in $(seq 1 300); do curl -sf "localhost:$1/health" >/dev/null && return 0; sleep 1; done; return 1; }

run_suite() {  # $1 = label, $2 = port
  local b="uv run python bench/bench_serving.py --base-url http://localhost:$2 --label $1 --output $OUT"
  $b --num-prompts 256 --input-len 512 --output-len 128                       # max throughput (burst)
  for rate in 4 8 16 32; do
    $b --num-prompts 256 --input-len 512 --output-len 128 --request-rate $rate --range-ratio 0.5
  done
  $b --num-prompts 256 --input-len 1024 --shared-prefix-len 768 --output-len 128 --request-rate 16
}

echo "== kvserve =="
uv run kvserve serve --port 8000 --model "$MODEL" --attention-backend triton \
  --kv-cache-memory-gb "$KV_GB" --max-num-seqs "$MAX_SEQS" --max-num-batched-tokens "$MAX_TOKENS" &
PID=$!; trap 'kill $PID 2>/dev/null || true' EXIT
wait_healthy 8000; run_suite kvserve 8000; kill $PID; wait $PID 2>/dev/null || true

echo "== vLLM =="
/workspace/vllm-env/bin/vllm serve "$MODEL" --port 8001 --max-num-seqs "$MAX_SEQS" \
  --max-num-batched-tokens "$MAX_TOKENS" --max-model-len 4096 --enable-prefix-caching \
  --kv-cache-memory-bytes "$((KV_GB * 1024 * 1024 * 1024))" &
PID=$!
wait_healthy 8001; run_suite vllm 8001; kill $PID; wait $PID 2>/dev/null || true

uv run python bench/summarize.py "$OUT"
