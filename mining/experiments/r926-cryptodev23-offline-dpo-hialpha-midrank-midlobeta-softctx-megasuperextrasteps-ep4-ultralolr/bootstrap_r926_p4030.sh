#!/usr/bin/env bash
set -euo pipefail
LOG=/root/logs/bootstrap_r926_p4030.log
mkdir -p /root/logs /root/hf /root/affine_data /root/mining_src /root/r926
exec > >(tee -a "$LOG") 2>&1
echo "[bootstrap-r926-p4030] $(date -u +%Y-%m-%dT%H:%M:%SZ) start host=$(hostname)"
if [[ -f /root/mine.env ]]; then set -a; source /root/mine.env; set +a; fi
export HF_HOME=${HF_HOME:-/root/hf}
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}
export HF_XET_HIGH_PERFORMANCE=${HF_XET_HIGH_PERFORMANCE:-1}
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export HF_TOKEN
[[ -n "${HF_TOKEN:-}" ]] || { echo FATAL missing HF_TOKEN; exit 1; }
nvidia-smi -L || true
EXP=r926-cryptodev23-offline-dpo-hialpha-midrank-midlobeta-softctx-megasuperextrasteps-ep4-ultralolr
test -s /root/mining_src/$EXP/dpo_duel_reason.jsonl
test -f /root/mining_src/$EXP/train_dpo.py
test -f /root/mining_src/$EXP/merge_lora.py
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
if [[ ! -d /root/venv ]]; then uv venv /root/venv --python 3.12; fi
source /root/venv/bin/activate
uv pip install \
  "torch==2.11.0" "transformers==5.14.1" "vllm==0.22.1" \
  "peft" "accelerate" "httpx" "huggingface_hub[hf_transfer]" \
  "hf_transfer" "safetensors" "numpy" "scipy" \
  2>&1 | tee /root/logs/pip_r926_p4030.log | tail -40
python - <<'PY'
import torch, transformers, vllm, peft
print("[bootstrap-r926-p4030] VERSIONS", torch.__version__, transformers.__version__, vllm.__version__, peft.__version__)
assert vllm.__version__.startswith("0.22.1")
assert transformers.__version__.startswith("5.14")
PY
python - <<'PY'
import os
from huggingface_hub import snapshot_download
token=os.environ["HF_TOKEN"]
repo="cryptoDev23/Affine-5Dku3dYp9j-hk8161"
rev="55b7ffe003d078a8a131673f677b2584548a502e"
print("[bootstrap-r926-p4030] DOWNLOAD cryptoDev start", repo, rev, flush=True)
path=snapshot_download(repo, revision=rev, token=token)
print("[bootstrap-r926-p4030] DOWNLOAD cryptoDev done", path, flush=True)
open("/root/logs/cryptodev_dl.done","w").write(path+"\n")
PY
BASE=/root/hf/hub/models--cryptoDev23--Affine-5Dku3dYp9j-hk8161/snapshots/55b7ffe003d078a8a131673f677b2584548a502e
test -e "$BASE/config.json"
cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl /root/r926/dpo_duel_reason.jsonl
chmod +x /root/mining_src/$EXP/*.sh
nohup bash /root/mining_src/$EXP/lean_train_h100_gpus01_p4030.sh \
  >/root/logs/p4030_r926_lean_train.outer.nohup 2>&1 &
echo $! >/root/logs/p4030_r926_lean_train.outer.pid
nohup bash /root/mining_src/$EXP/wait_r926_train_then_merge_p4030.sh \
  >/root/logs/p4030_r926_wait_merge.nohup 2>&1 &
echo $! >/root/logs/p4030_r926_wait_merge.pid
sleep 5
echo "R926_TRAIN_PID=$(cat /root/logs/r926_train.pid 2>/dev/null || echo pending)"
tail -20 /root/logs/r926_lean_warm.log 2>/dev/null || true
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4030_r926_armed.done
echo "[bootstrap-r926-p4030] ARMED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
