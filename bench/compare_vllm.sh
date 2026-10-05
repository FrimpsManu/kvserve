#!/usr/bin/env bash
# Like-for-like serving benchmark: kvserve vs. vLLM, same GPU, model, scheduler limits and prompts.
#   bash bench/compare_vllm.sh                       (run on the GPU pod after scripts/pod_setup.sh --vllm)
#   SYSTEMS=vllm OUT=results/x.jsonl bash bench/compare_vllm.sh   (one system, append to existing results)
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/workspace/hf}
MODEL=${MODEL:-unsloth/Llama-3.2-1B-Instruct}
KV_GB=${KV_GB:-8}
MAX_SEQS=${MAX_SEQS:-128}
MAX_TOKENS=${MAX_TOKENS:-2048}
SYSTEMS=${SYSTEMS:-"kvserve vllm"}
KVSERVE_PORT=${KVSERVE_PORT:-8000}
VLLM_PORT=${VLLM_PORT:-8100}  # RunPod images run nginx on 8001
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')
OUT=${OUT:-results/serving_${GPU}_$(date +%Y%m%d_%H%M%S).jsonl}
mkdir -p results

port_free() {
  if ss -ltn "sport = :$1" | grep -q LISTEN; then echo "port $1 is already in use" >&2; return 1; fi
}

# Ready = /v1/models answers with an OpenAI model list (not just any HTTP 200).
wait_ready() {  # $1 = port, $2 = server pid
  for _ in $(seq 1 600); do
    kill -0 "$2" 2>/dev/null || { echo "server exited during startup" >&2; return 1; }
    curl -sf "localhost:$1/v1/models" | python3 -c "import json,sys; json.load(sys.stdin)['data'][0]['id']" \
      2>/dev/null && return 0
    sleep 1
  done
  echo "server on port $1 not ready after 600s" >&2
  return 1
}

run_suite() {  # $1 = label, $2 = port
  local b="uv run python bench/bench_serving.py --base-url http://localhost:$2 --label $1 --output $OUT"
  $b --num-prompts 256 --input-len 512 --output-len 128                       # max throughput (burst)
  for rate in 4 8 16 32; do
    $b --num-prompts 256 --input-len 512 --output-len 128 --request-rate $rate --range-ratio 0.5
  done
  $b --num-prompts 256 --input-len 1024 --shared-prefix-len 768 --output-len 128 --request-rate 16
}

PID=""
trap '[[ -n "$PID" ]] && kill $PID 2>/dev/null || true' EXIT

for system in $SYSTEMS; do
  echo "== $system =="
  case $system in
    kvserve)
      port=$KVSERVE_PORT; port_free "$port"
      uv run kvserve serve --port "$port" --model "$MODEL" --attention-backend triton \
        --kv-cache-memory-gb "$KV_GB" --max-num-seqs "$MAX_SEQS" --max-num-batched-tokens "$MAX_TOKENS" &
      ;;
    vllm)
      port=$VLLM_PORT; port_free "$port"
      /workspace/vllm-env/bin/vllm serve "$MODEL" --port "$port" --max-num-seqs "$MAX_SEQS" \
        --max-num-batched-tokens "$MAX_TOKENS" --max-model-len 4096 --enable-prefix-caching \
        --kv-cache-memory-bytes "$((KV_GB * 1024 * 1024 * 1024))" &
      ;;
    *) echo "unknown system $system" >&2; exit 1 ;;
  esac
  PID=$!
  wait_ready "$port" "$PID"
  run_suite "$system" "$port"
  kill "$PID"; wait "$PID" 2>/dev/null || true; PID=""
done

uv run python bench/summarize.py "$OUT"
