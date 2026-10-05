#!/usr/bin/env bash
# Copy the working tree to a pod (run on your laptop).
#   scripts/sync_to_pod.sh root@<ip> <port>
# Results are pulled back from <pod>:/workspace/kvserve/results.
set -euo pipefail
host=${1:?usage: sync_to_pod.sh user@host port}
port=${2:?usage: sync_to_pod.sh user@host port}
key=${KVSERVE_SSH_KEY:-$HOME/.ssh/runpod_ed25519}
ssh_cmd="ssh -i $key -p $port -o StrictHostKeyChecking=accept-new"
cd "$(dirname "$0")/.."
rsync -az --delete -e "$ssh_cmd" --exclude .venv --exclude results --exclude __pycache__ --exclude .git \
  ./ "$host:/workspace/kvserve/"
mkdir -p results
rsync -az -e "$ssh_cmd" "$host:/workspace/kvserve/results/" results/ 2>/dev/null || true
echo "synced to $host:/workspace/kvserve"
