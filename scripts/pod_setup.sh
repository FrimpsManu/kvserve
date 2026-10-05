#!/usr/bin/env bash
# One-time setup on a fresh GPU pod (RunPod PyTorch template or any Ubuntu + NVIDIA driver box).
#   bash scripts/pod_setup.sh            # kvserve only
#   bash scripts/pod_setup.sh --vllm     # also install vLLM (separate venv) for comparisons
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_HOME=${HF_HOME:-/workspace/hf}  # persistent volume on RunPod
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
uv sync
uv run python -c "import torch, triton; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'triton', triton.__version__, torch.cuda.get_device_name())"
uv run python -c "from kvserve.config import resolve_model_path, DEFAULT_MODEL; print(resolve_model_path(DEFAULT_MODEL))"

if [[ "${1:-}" == "--vllm" ]]; then
  uv venv /workspace/vllm-env --python 3.12
  # ninja: vLLM JIT-compiles some kernels (e.g. FlashInfer) at startup.
  VIRTUAL_ENV=/workspace/vllm-env uv pip install vllm ninja
  /workspace/vllm-env/bin/python -c "import vllm; print('vllm', vllm.__version__)"
fi
echo "setup done. export HF_HOME=$HF_HOME"
