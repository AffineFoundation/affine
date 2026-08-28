#!/usr/bin/env bash
# Track P pod setup (8xH200, sm_90). Mirrors Track B's layout:
#   /dshare          big-disk shared dir, mounted into every vLLM container
#   /dshare/hf       HF cache (/ has 2.1T free; /root is a small 1.3T gocryptfs)
#   /root/vllmenv    hf download CLI only (serving is docker)
#   /root/trainenv   torch+transformers+peft (driver, D LoRA train, G SFT)
# Models: teacher/discriminator base Qwen/Qwen3.8-27B, generator Qwen/Qwen3-4B.
set -uo pipefail
export HF_HOME=/dshare/hf
export HF_TOKEN="$(cat /root/.hf_token)"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p /root/work /dshare/hf /dshare/gad

log() { echo "[setup $(date -u +%T)] $*"; }

command -v uv >/dev/null 2>&1 || (curl -LsSf https://astral.sh/uv/install.sh | sh)
export PATH="$HOME/.local/bin:$PATH"

log "venv: vllmenv (hf cli)"
uv venv /root/vllmenv --python 3.12
VIRTUAL_ENV=/root/vllmenv uv pip install -q "huggingface_hub[hf_transfer]" || log "WARN hf install failed"

# 27B download is the long pole (~54GB) -- start immediately
log "downloading Qwen/Qwen3.8-27B (background)"
nohup /root/vllmenv/bin/hf download Qwen/Qwen3.8-27B > /root/work/dl_27b.log 2>&1 &
DL27=$!

log "docker pull vllm images (background)"
nohup docker pull vllm/vllm-openai:v0.10.1.1 > /root/work/pull_v0101.log 2>&1 &
nohup docker pull vllm/vllm-openai:v0.11.0 > /root/work/pull_v0110.log 2>&1 &

log "venv: trainenv"
uv venv /root/trainenv --python 3.12
VIRTUAL_ENV=/root/trainenv uv pip install -q torch transformers peft requests numpy safetensors accelerate || log "WARN trainenv install failed"

log "downloading Qwen/Qwen3-4B"
/root/vllmenv/bin/hf download Qwen/Qwen3-4B > /root/work/dl_4b.log 2>&1

log "waiting for 27B download"
wait $DL27 || log "WARN 27B download exited nonzero"

log "verify"
/root/trainenv/bin/python -c "import torch, transformers, peft; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
du -sh /dshare/hf
echo "SETUP_DONE"
