#!/usr/bin/env bash
# Factory pod setup: standard harness for isomorphism-factory arms.
#   teacher Qwen/Qwen3.6-35B-A3B   (0.32 on fastbench seed-0 n=25)
#   miner   Qwen/Qwen3-8B
# No benchenv / no panel images: SWE benching happens on the fastbench box.
# Container-visible files live under $D (=/dshare when present: on some
# providers /root is gocryptfs FUSE and docker bind mounts out of it fail).
set -uo pipefail
D=/dshare; [ -d /dshare ] || { D=/root/dshare; mkdir -p "$D"; }
echo "$D" > /root/.dshare_path
export HF_HOME="$D/hf"
export HF_TOKEN="$(cat /root/.hf_token)"
mkdir -p /root/work "$D/hf" "$D/factory" "$D/ckpt"

log() { echo "[setup $(date -u +%T)] $*"; }

command -v uv >/dev/null 2>&1 || (curl -LsSf https://astral.sh/uv/install.sh | sh)
export PATH="$HOME/.local/bin:$PATH"

log "venv: vllmenv (hf cli only)"
uv venv /root/vllmenv --python 3.12
VIRTUAL_ENV=/root/vllmenv uv pip install -q "huggingface_hub[hf_transfer]" || log "WARN hf install failed"

log "downloading teacher Qwen3.6-35B-A3B (background, long pole)"
nohup /root/vllmenv/bin/hf download Qwen/Qwen3.6-35B-A3B > /root/work/dl_teacher.log 2>&1 &
DLT=$!

log "docker pull vllm/vllm-openai:latest (background)"
nohup docker pull vllm/vllm-openai:latest > /root/work/pull_vllm.log 2>&1 &

log "venv: trainenv"
uv venv /root/trainenv --python 3.12
VIRTUAL_ENV=/root/trainenv uv pip install -q torch transformers peft requests numpy safetensors accelerate || log "WARN trainenv install failed"

log "downloading Qwen3-8B miner"
/root/vllmenv/bin/hf download Qwen/Qwen3-8B > /root/work/dl_8b.log 2>&1 || log "WARN 8B download failed"

# fleet key: pods scp merged weights to the bench box and data to each other
mkdir -p /root/.ssh
if [ -f /root/factory_key ]; then
  mv /root/factory_key /root/.ssh/factory_key; chmod 600 /root/.ssh/factory_key
fi
if [ -f /root/factory_key.pub ]; then
  cat /root/factory_key.pub >> /root/.ssh/authorized_keys
fi

log "waiting for teacher download"
wait $DLT || log "WARN teacher download exited nonzero"

log "verify"
/root/trainenv/bin/python -c "import torch, transformers, peft; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count(), 'gpus')" || log "WARN trainenv broken"
du -sh "$D/hf"
echo "SETUP_DONE"
