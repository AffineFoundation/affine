#!/usr/bin/env bash
# Blank H200 mine-r1214 → R1218 MidCtx MidRank Midβ Mega HiLR train.
# Never pkill -f.
set -euo pipefail
LOG=/root/logs/bootstrap_r1218_p4326.log
mkdir -p /root/logs /root/hf /root/affine_data /root/mining_src /root/r1218
exec > >(tee -a "$LOG") 2>&1
echo "[bootstrap-r1218-p4326] $(date -u +%Y-%m-%dT%H:%M:%SZ) start host=$(hostname)"
if [[ -f /root/mine.env ]]; then set -a; source /root/mine.env; set +a; fi
export HF_HOME=${HF_HOME:-/root/hf}
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}
export HF_XET_HIGH_PERFORMANCE=${HF_XET_HIGH_PERFORMANCE:-1}
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export HF_TOKEN
[[ -n "${HF_TOKEN:-}" ]] || { echo FATAL missing HF_TOKEN; exit 1; }
nvidia-smi -L || true
EXP=r1218-vera-offline-dpo-hialpha-midrank-midbeta-midctx-megasuperextrasteps-ep4-hilr
test -f /root/mining_src/$EXP/train_dpo.py
test -f /root/mining_src/$EXP/merge_lora.py
test -s /root/mining_src/$EXP/dpo_duel_reason.jsonl
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
  2>&1 | tee /root/logs/pip_r1218_p4326.log | tail -40
python - <<'PY'
import torch, transformers, vllm, peft
print("[bootstrap-r1218-p4326] VERSIONS", torch.__version__, transformers.__version__, vllm.__version__, peft.__version__)
assert vllm.__version__.startswith("0.22.1")
assert transformers.__version__.startswith("5.14")
PY
python - <<'PY'
import os
from huggingface_hub import snapshot_download
token=os.environ["HF_TOKEN"]
repo="vera6/affine-5g4yy75zuz-t6"
rev="8e3f1695e058837ed80fec3238ff439fdc2d0f0e"
print("[bootstrap-r1218-p4326] DOWNLOAD king start", repo, rev, flush=True)
path=snapshot_download(repo, revision=rev, token=token)
print("[bootstrap-r1218-p4326] DOWNLOAD king done", path, flush=True)
open("/root/logs/vera_king_dl.done","w").write(path+"\n")
PY
BASE=/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/8e3f1695e058837ed80fec3238ff439fdc2d0f0e
test -e "$BASE/config.json"
cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl /root/r1218/dpo_duel_reason.jsonl
chmod +x /root/mining_src/$EXP/*.sh
nohup bash /root/mining_src/$EXP/lean_train_h200_gpus01_p4326.sh \
  >/root/logs/p4326_r1218_lean_train.outer.nohup 2>&1 &
echo $! >/root/logs/p4326_r1218_lean_train.outer.pid
sleep 8
echo "R1218_TRAIN_PID=$(cat /root/logs/r1218_train.pid 2>/dev/null || echo pending)"
echo "R1218_OUTER=$(cat /root/logs/p4326_r1218_lean_train.outer.pid)"
tail -30 /root/logs/r1218_lean_warm.log 2>/dev/null || true
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4326_r1218_armed.done
echo "[bootstrap-r1218-p4326] ARMED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
