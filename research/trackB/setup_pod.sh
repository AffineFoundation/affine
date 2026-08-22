#!/usr/bin/env bash
# Track B pod setup: 8xH200 (sm_90, mature wheels -- no CUDA_HOME gymnastics).
# Three venvs so vllm / training / bench deps can never fight:
#   /root/vllmenv   vllm serving
#   /root/trainenv  torch+transformers+peft training (driver + G SFT)
#   /root/benchenv  mini-swe-agent + swebench fork harness
# Models pre-downloaded to /root/hf so server starts are disk-bound.
set -uo pipefail
export HF_HOME=/root/hf
export HF_TOKEN="$(cat /root/.hf_token)"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p /root/work /root/hf /root/bench /root/work/gad

log() { echo "[setup $(date -u +%T)] $*"; }

command -v uv >/dev/null 2>&1 || (curl -LsSf https://astral.sh/uv/install.sh | sh)
export PATH="$HOME/.local/bin:$PATH"

log "venv: vllmenv"
uv venv /root/vllmenv --python 3.12
VIRTUAL_ENV=/root/vllmenv uv pip install -q vllm "huggingface_hub[hf_transfer]" || log "WARN vllm install failed"

# teacher download starts as early as possible -- it is the long pole (~65GB)
log "downloading Qwen3-32B (background)"
nohup /root/vllmenv/bin/hf download Qwen/Qwen3-32B > /root/work/dl_32b.log 2>&1 &
DL32=$!

log "venv: trainenv"
uv venv /root/trainenv --python 3.12
VIRTUAL_ENV=/root/trainenv uv pip install -q torch transformers peft requests numpy || log "WARN trainenv install failed"

log "venv: benchenv"
uv venv /root/benchenv --python 3.12
VIRTUAL_ENV=/root/benchenv uv pip install -q mini-swe-agent datasets pyarrow pyyaml requests || log "WARN benchenv install failed"
VIRTUAL_ENV=/root/benchenv uv pip install -q "swebench @ git+https://github.com/SWE-rebench/SWE-bench-fork" || log "WARN swebench install failed"

log "downloading Qwen3-8B and Qwen3-4B"
/root/vllmenv/bin/hf download Qwen/Qwen3-8B > /root/work/dl_8b.log 2>&1
/root/vllmenv/bin/hf download Qwen/Qwen3-4B > /root/work/dl_4b.log 2>&1

log "waiting for 32B download"
wait $DL32 || log "WARN 32B download exited nonzero"

log "verify"
/root/vllmenv/bin/python -c "import vllm; print('vllm', vllm.__version__)"
/root/trainenv/bin/python -c "import torch, transformers, peft; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
/root/benchenv/bin/python -c "import swebench, minisweagent, yaml; print('bench ok')"
echo "SETUP_DONE"
