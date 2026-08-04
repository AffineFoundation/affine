#!/bin/bash
# Bootstrap affv-e6 for wave-5 more kings (IV/XLIII/XLVII/IX/XLIV/VI).
set -euo pipefail
export HF_HOME=/root/hf
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p /root/logs /root/data /root/results /root/hf /root/harness

if [ -f /root/.env.hf ]; then
  # shellcheck disable=SC1091
  source /root/.env.hf
fi
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
if [ ! -x /root/venv/bin/python ]; then
  uv venv /root/venv --python 3.12
fi
# shellcheck disable=SC1091
source /root/venv/bin/activate
uv pip install -U vllm httpx transformers accelerate sentencepiece protobuf scipy

python - <<'PY'
from huggingface_hub import snapshot_download
import os
tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
repos = [
    "zai-org/GLM-4.5-Air-FP8",
    "dendriteholdings/albedo-qwen3.6-35b-king-genesis",
    "dendriteholdings/albedo-qwen3.6-35b-king-IV",
    "dendriteholdings/albedo-qwen3.6-35b-king-XLIII",
    "dendriteholdings/albedo-qwen3.6-35b-king-XLVII",
    "dendriteholdings/albedo-qwen3.6-35b-king-IX",
    "dendriteholdings/albedo-qwen3.6-35b-king-XLIV",
    "dendriteholdings/albedo-qwen3.6-35b-king-VI",
]
for repo in repos:
    print("START", repo, flush=True)
    snapshot_download(repo, token=tok)
    print("DONE", repo, flush=True)
print("ALL_DONE", flush=True)
PY
