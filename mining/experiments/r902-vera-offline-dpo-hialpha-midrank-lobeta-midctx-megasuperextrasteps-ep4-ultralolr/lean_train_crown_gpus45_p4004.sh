#!/usr/bin/env bash
set -euo pipefail
exec >/root/logs/r902_lean_warm.log 2>&1
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=4,5 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[r902-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) start GPUs 4,5"
BASE=/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/8e3f1695e058837ed80fec3238ff439fdc2d0f0e
EXP=r902-vera-offline-dpo-hialpha-midrank-lobeta-midctx-megasuperextrasteps-ep4-ultralolr
mkdir -p /root/r902 /root/logs /root/affine_data /root/mining_src/s4-h138-f43-tok-dpo-l2 /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r902/dpo_duel_reason.jsonl
test -s "$DATA"
n=$(wc -l <"$DATA"); echo "[r902-lean] data_lines=$n"; test "$n" -ge 200
cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
if [[ -f /root/logs/r902_train.pid ]]; then
  old=$(cat /root/logs/r902_train.pid 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "FATAL r902 already alive pid=$old"; exit 1
  fi
fi
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo "FATAL GPUs 4,5 busy used=$used"; exit 1; }
rm -rf /root/r902/train; mkdir -p /root/r902/train; : >/root/logs/r902_train.nohup
chmod +x /root/mining_src/$EXP/*.sh
export BASE CUDA_VISIBLE_DEVICES=4,5
export R902_LR=5e-7 R902_MAX_STEPS=19200 R902_LORA_R=32 R902_LORA_ALPHA=128 R902_BETA=0.02 R902_MAX_LEN=8192 R902_EPOCHS=4
export SRC=/root/mining_src/s4-h138-f43-tok-dpo-l2 OUT=/root/r902 DATA=/root/r902/dpo_duel_reason.jsonl LOG=/root/logs/r902_train.nohup SKIP_LOCAL_TKC=1
bash /root/mining_src/$EXP/start_r902.sh
echo "[r902-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) TRAIN launched pid=$(cat /root/logs/r902_train.pid)"
