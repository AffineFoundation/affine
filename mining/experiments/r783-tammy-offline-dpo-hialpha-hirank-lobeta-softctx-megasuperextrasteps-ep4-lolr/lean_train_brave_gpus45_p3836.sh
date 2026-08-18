#!/usr/bin/env bash
# p3836: R783 tammy Soft HiRank LoBeta SoftCtx Mega on brave GPUs 4,5 (fill idle after R767 MERGE)
set -euo pipefail
exec >/root/logs/r783_lean_warm.log 2>&1
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=4,5 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[r783-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) start GPUs 4,5"
BASE=/root/hf/hub/models--tammyfritz--Affine-5hmwhnfbix-tammy2/snapshots/7e5fd5f87e82606c32d59c3d2350e3ddfe49c4b5
EXP=r783-tammy-offline-dpo-hialpha-hirank-lobeta-softctx-megasuperextrasteps-ep4-lolr
mkdir -p /root/r783 /root/logs /root/affine_data \
  /root/mining_src/s4-h138-f43-tok-dpo-l2 \
  /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r783/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  if [[ -s /root/mining_src/$EXP/dpo_duel_reason.jsonl ]]; then
    cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r781/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r781/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r767/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r767/dpo_duel_reason.jsonl "$DATA"
  else
    echo "FATAL missing SoftCtx data"; exit 1
  fi
fi
n=$(wc -l <"$DATA"); echo "[r783-lean] data_lines=$n"; test "$n" -ge 200
cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
if [[ -f /root/logs/r783_train.pid ]]; then
  old=$(cat /root/logs/r783_train.pid 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "FATAL r783 already alive pid=$old"; exit 1
  fi
fi
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
  echo "[r783-lean] wait VRAM4+5 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo "FATAL GPUs 4,5 busy"; exit 1; }
rm -rf /root/r783/train; mkdir -p /root/r783/train; : >/root/logs/r783_train.nohup
chmod +x /root/mining_src/$EXP/*.sh
export BASE
export CUDA_VISIBLE_DEVICES=4,5 R783_LR=1e-6 R783_MAX_STEPS=19200 R783_LORA_R=64 R783_LORA_ALPHA=128 R783_BETA=0.02 R783_MAX_LEN=12288 R783_EPOCHS=4
export SRC=/root/mining_src/s4-h138-f43-tok-dpo-l2 OUT=/root/r783 DATA=/root/r783/dpo_duel_reason.jsonl LOG=/root/logs/r783_train.nohup SKIP_LOCAL_TKC=1
bash /root/mining_src/$EXP/start_r783.sh
echo "[r783-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) TRAIN launched pid=$(cat /root/logs/r783_train.pid) BASE=$BASE"
