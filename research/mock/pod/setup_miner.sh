#!/usr/bin/env bash
# Miner box setup (8xB300 Blackwell -> Docker vLLM :latest ONLY, Track A's route).
# Layout: /dshare big-disk shared dir (HF cache /dshare/hf), /root/work code,
# /dshare/koth miner loop state.
set -uo pipefail
export HF_HOME=/dshare/hf
export HF_TOKEN="$(cat /root/.hf_token)"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p /root/work /dshare/hf /dshare/koth /dshare/koth/published

log() { echo "[setup $(date -u +%T)] $*"; }

command -v uv >/dev/null 2>&1 || (curl -LsSf https://astral.sh/uv/install.sh | sh)
export PATH="$HOME/.local/bin:$PATH"

uv venv /root/vllmenv --python 3.12
VIRTUAL_ENV=/root/vllmenv uv pip install -q "huggingface_hub[hf_transfer]" || log "WARN hf cli install failed"

log "downloading Qwen/Qwen3.6-35B-A3B (background)"
nohup /root/vllmenv/bin/hf download Qwen/Qwen3.6-35B-A3B > /root/work/dl_35b.log 2>&1 &
DL35=$!
log "docker pull vllm:latest (background)"
nohup docker pull vllm/vllm-openai:latest > /root/work/pull_latest.log 2>&1 &

uv venv /root/trainenv --python 3.12
VIRTUAL_ENV=/root/trainenv uv pip install -q torch transformers peft requests numpy safetensors accelerate || log "WARN trainenv install failed"

log "downloading Qwen/Qwen3.8-27B"
/root/vllmenv/bin/hf download Qwen/Qwen3.8-27B > /root/work/dl_27b.log 2>&1
wait $DL35 || log "WARN 35B download exited nonzero"

/root/trainenv/bin/python -c "import torch, transformers, peft; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
du -sh /dshare/hf
echo "SETUP_DONE"
