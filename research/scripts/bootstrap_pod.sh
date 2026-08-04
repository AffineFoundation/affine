#!/bin/bash
# Bootstrap a fresh Lium pod for affv harness experiments.
set -euo pipefail
export HF_HOME=/root/hf
export HF_TOKEN="${HF_TOKEN:-}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

apt-get update -qq
apt-get install -y -qq git curl wget build-essential 2>/dev/null || true

python3 -m venv /root/venv
source /root/venv/bin/activate
pip install -q -U pip wheel
# Match e1 stack closely; CUDA 12/13 wheels from vLLM
pip install -q "vllm>=0.8" httpx transformers accelerate

mkdir -p /root/harness /root/data /root/results /root/logs /root/hf /root/scripts
echo "bootstrap ok: $(python -c 'import vllm; print(vllm.__version__)')"
