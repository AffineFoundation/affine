#!/usr/bin/env bash
# Track T pod setup (8xB300 brave-wolf-83). Mirrors Track S/M conventions:
#   /dshare          big-disk dir on / (overlay 7T), mounted into containers
#   /dshare/hf       HF cache (/root is lium-cipher FUSE; docker bind-mounts
#                    out of it fail -> everything containers see lives here)
#   /root/trainenv   torch+transformers+peft (driver, D train, G SFT)
#   /root/vllmenv    hf download CLI only (serving is docker :latest, the
#                    Blackwell route -- pip vLLM cannot JIT for sm_103a)
#   /root/benchenv   mini-swe-agent + SWE-rebench fork (SWE panels)
# Models: teacher/judge Qwen/Qwen3.8-27B; student (raw, GAN-only)
# dendriteholdings/albedo-qwen3.6-35b-king-genesis (35B-A3B miner arch).
set -uo pipefail
export HF_HOME=/dshare/hf
export HF_TOKEN="$(cat /root/.hf_token)"
export HF_HUB_ENABLE_HF_TRANSFER=1
STUDENT="dendriteholdings/albedo-qwen3.6-35b-king-genesis"
mkdir -p /root/work /dshare/hf /dshare/trackT /dshare/bench
date +%s > /dshare/trackT/pod_t0 2>/dev/null || true

log() { echo "[setup $(date -u +%T)] $*"; }

command -v uv >/dev/null 2>&1 || (curl -LsSf https://astral.sh/uv/install.sh | sh)
export PATH="$HOME/.local/bin:$PATH"

log "venv: vllmenv (hf cli)"
uv venv /root/vllmenv --python 3.12
VIRTUAL_ENV=/root/vllmenv uv pip install -q "huggingface_hub[hf_transfer]" || log "WARN hf install failed"

log "downloading $STUDENT (background)"
nohup /root/vllmenv/bin/hf download "$STUDENT" > /root/work/dl_student.log 2>&1 &
DLS=$!
log "downloading Qwen/Qwen3.8-27B (background)"
nohup /root/vllmenv/bin/hf download Qwen/Qwen3.8-27B > /root/work/dl_27b.log 2>&1 &
DL27=$!

log "docker pull vllm/vllm-openai:latest (background)"
nohup docker pull vllm/vllm-openai:latest > /root/work/pull_latest.log 2>&1 &

log "venv: trainenv"
uv venv /root/trainenv --python 3.12
VIRTUAL_ENV=/root/trainenv uv pip install -q torch transformers peft requests numpy safetensors accelerate || log "WARN trainenv install failed"

log "venv: benchenv"
uv venv /root/benchenv --python 3.12
VIRTUAL_ENV=/root/benchenv uv pip install -q mini-swe-agent datasets pyarrow pyyaml requests || log "WARN benchenv install failed"
VIRTUAL_ENV=/root/benchenv uv pip install -q "swebench @ git+https://github.com/SWE-rebench/SWE-bench-fork" || log "WARN swebench install failed"
/root/benchenv/bin/python -c "import swebench, minisweagent, yaml; print('bench env ok')" || log "WARN bench env broken"

log "prepull panel images (background: proxy first, then full)"
nohup bash -c 'xargs -P 4 -n 1 docker pull < /root/work/panels/images_proxy.txt; xargs -P 4 -n 1 docker pull < /root/work/panels/images_full.txt' > /root/work/prepull.log 2>&1 &

log "waiting for model downloads"
wait $DL27 || log "WARN 27B download exited nonzero"
wait $DLS || log "WARN student download exited nonzero"

log "verify"
/root/trainenv/bin/python -c "import torch, transformers, peft; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
du -sh /dshare/hf
echo "SETUP_DONE"
