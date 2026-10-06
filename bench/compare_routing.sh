#!/usr/bin/env bash
# Routing experiment: N kvserve instances behind kvgateway, one run per routing policy.
# Instances restart for every policy so each starts with a cold prefix cache.
#
#   bash bench/compare_routing.sh                              # 2 instances, default workload
#   N=4 NGPUS=4 KV_GB=4 bash bench/compare_routing.sh          # one instance per GPU
#   POLICIES="round_robin prefix_affinity" BENCH_ARGS="..." bash bench/compare_routing.sh
#
# The default workload is many requests drawing from 16 long shared prefixes with Zipf
# popularity (a few hot "applications"), sized so that one instance's KV cache cannot
# hold every prefix: routing decides whether instances share the work or duplicate it.
set -euo pipefail
cd "$(dirname "$0")/.."

N=${N:-2}
NGPUS=${NGPUS:-0}                      # >0: pin instance i to GPU i % NGPUS
POLICIES=${POLICIES:-"round_robin least_loaded prefix_affinity"}
KV_GB=${KV_GB:-1}
EXTRA_SERVE_ARGS=${EXTRA_SERVE_ARGS:-}
BASE_PORT=${BASE_PORT:-8100}
GW_PORT=${GW_PORT:-9000}
BENCH_ARGS=${BENCH_ARGS:-"--num-prompts 256 --input-len 1024 --shared-prefix-len 896 --num-prefixes 16 \
  --prefix-dist zipf --output-len 64 --request-rate 8 --range-ratio 0.2"}
OUT=${OUT:-results/routing_$(date +%Y%m%d_%H%M%S).jsonl}
mkdir -p results

(cd gateway && go build -o bin/kvgateway ./cmd/kvgateway)

PIDS=()
cleanup() { for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done; wait 2>/dev/null || true; PIDS=(); }
trap cleanup EXIT

port_free() { ! (echo >"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }
wait_ready() {  # $1 = url, $2 = pid
  for _ in $(seq 1 900); do
    kill -0 "$2" 2>/dev/null || { echo "process for $1 exited during startup" >&2; return 1; }
    curl -sf "$1/health" >/dev/null && return 0
    sleep 1
  done
  echo "$1 not ready" >&2; return 1
}

# Written for bash 3.2 (macOS) as well: no empty-array expansion under `set -u`, no
# negative array indexes.
launch_backend() {  # $1 = index, $2 = port
  local log="results/.routing_backend_$1.log"
  if [[ $NGPUS -gt 0 ]]; then
    CUDA_VISIBLE_DEVICES=$(($1 % NGPUS)) uv run kvserve serve --port "$2" --kv-cache-memory-gb "$KV_GB" \
      $EXTRA_SERVE_ARGS > "$log" 2>&1 &
  else
    uv run kvserve serve --port "$2" --kv-cache-memory-gb "$KV_GB" $EXTRA_SERVE_ARGS > "$log" 2>&1 &
  fi
}

for policy in $POLICIES; do
  echo "== $policy ($N instances) =="
  urls=()
  for i in $(seq 0 $((N - 1))); do
    port=$((BASE_PORT + i))
    port_free "$port" || { echo "port $port in use" >&2; exit 1; }
    launch_backend "$i" "$port"
    PIDS+=($!)
    urls+=("http://127.0.0.1:$port")
  done
  for i in "${!urls[@]}"; do wait_ready "${urls[$i]}" "${PIDS[$i]}"; done

  backends=$(IFS=,; echo "${urls[*]}")
  gateway/bin/kvgateway --listen ":$GW_PORT" --backends "$backends" --policy "$policy" \
    > "results/.routing_gateway.log" 2>&1 &
  gw_pid=$!
  PIDS+=($gw_pid)
  wait_ready "http://127.0.0.1:$GW_PORT" "$gw_pid"

  uv run python bench/bench_serving.py --base-url "http://127.0.0.1:$GW_PORT" --label "$policy" \
    --stats-urls "$backends" --output "$OUT" $BENCH_ARGS
  cleanup
  sleep 3  # let ports and GPU memory free up
done

uv run python - "$OUT" <<'EOF'
import json, sys
print(f"\n| Policy | Prefix-cache hit | Output tok/s | TTFT p50 / p99 (ms) | TPOT p50 (ms) | Goodput (req/s) |")
print("|---|---|---|---|---|---|")
for line in open(sys.argv[1]):
    r = json.loads(line)
    pc = r.get("prefix_cache", {}).get("hit_rate", 0)
    t, p = r["ttft_ms"], r["tpot_ms"]
    print(f"| {r['label']} | {100 * pc:.1f}% | {r['output_throughput']:.0f} | {t['p50']:.0f} / {t['p99']:.0f} "
          f"| {p['p50']:.1f} | {r['goodput']:.2f} |")
EOF
