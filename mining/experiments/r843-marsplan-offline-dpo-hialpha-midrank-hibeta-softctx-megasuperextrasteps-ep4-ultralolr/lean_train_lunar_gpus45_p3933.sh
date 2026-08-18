#!/usr/bin/env bash
# p3933: R843 SoftCtx MidRank HiBeta UltraLoLR after R837 SoftCtx LoBeta REFUTE ~0.07×
set -euo pipefail
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=4,5 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[r843-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) start GPUs 4,5"
BASE=/root/hf/hub/models--marsplan0624--affine-5gedzafcvg-queen/snapshots/556d02a2adfa9bd42a02de3c766f98be7e44ca46
EXP=r843-marsplan-offline-dpo-hialpha-midrank-hibeta-softctx-megasuperextrasteps-ep4-ultralolr
mkdir -p /root/r843 /root/logs /root/affine_data \
  /root/mining_src/s4-h138-f43-tok-dpo-l2 \
  /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r843/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  if [[ -s /root/r837/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r837/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r842/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r842/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r789/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r789/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r808/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r808/dpo_duel_reason.jsonl "$DATA"
  else
    echo "FATAL missing SoftCtx DPO data"; exit 1
  fi
fi
n=$(wc -l <"$DATA"); echo "[r843-lean] data_lines=$n"; test "$n" -ge 200
cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
if [[ -f /root/logs/r843_train.pid ]]; then
  old=$(cat /root/logs/r843_train.pid 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "FATAL r843 already alive pid=$old"; exit 1
  fi
fi
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
  echo "[r843-lean] wait VRAM4+5 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo "FATAL GPUs 4,5 busy"; exit 1; }
rm -rf /root/r843/train; mkdir -p /root/r843/train; : >/root/logs/r843_train.nohup
chmod +x /root/mining_src/$EXP/*.sh
export BASE
export CUDA_VISIBLE_DEVICES=4,5 R843_LR=5e-7 R843_MAX_STEPS=19200 R843_LORA_R=32 R843_LORA_ALPHA=128 R843_BETA=0.3 R843_MAX_LEN=12288 R843_EPOCHS=4
export SRC=/root/mining_src/s4-h138-f43-tok-dpo-l2 OUT=/root/r843 DATA=/root/r843/dpo_duel_reason.jsonl LOG=/root/logs/r843_train.nohup SKIP_LOCAL_TKC=1
bash /root/mining_src/$EXP/start_r843.sh
echo "[r843-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) TRAIN launched pid=$(cat /root/logs/r843_train.pid) BASE=$BASE"
