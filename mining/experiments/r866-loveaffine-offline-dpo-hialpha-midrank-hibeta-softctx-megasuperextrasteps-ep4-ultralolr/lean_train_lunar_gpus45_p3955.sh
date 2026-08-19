#!/usr/bin/env bash
# p3955: R866 loveaffine SoftCtx MidRank HiBeta UltraLoLR after R843 marsplan REFUTE ~-0.51×
set -euo pipefail
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=4,5 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[r866-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) start GPUs 4,5"
BASE=/root/hf/hub/models--justice101--affine-5dz2gkonkn-loveaffine/snapshots/b21914bd6476c93f896328954a2f74db4b63efba
EXP=r866-loveaffine-offline-dpo-hialpha-midrank-hibeta-softctx-megasuperextrasteps-ep4-ultralolr
mkdir -p /root/r866 /root/logs /root/affine_data \
  /root/mining_src/s4-h138-f43-tok-dpo-l2 \
  /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r866/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  if [[ -s /root/r843/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r843/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r837/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r837/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r842/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r842/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r789/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r789/dpo_duel_reason.jsonl "$DATA"
  else
    echo "FATAL missing SoftCtx Soft Mid Mid Soft DPO data"; exit 1
  fi
fi
n=$(wc -l <"$DATA"); echo "[r866-lean] data_lines=$n"; test "$n" -ge 200
cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
if [[ -f /root/logs/r866_train.pid ]]; then
  old=$(cat /root/logs/r866_train.pid 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "FATAL r866 already alive pid=$old"; exit 1
  fi
fi
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
  echo "[r866-lean] wait VRAM4+5 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo "FATAL GPUs 4,5 busy"; exit 1; }
rm -rf /root/r866/train; mkdir -p /root/r866/train; : >/root/logs/r866_train.nohup
chmod +x /root/mining_src/$EXP/*.sh
export BASE
export CUDA_VISIBLE_DEVICES=4,5 R866_LR=5e-7 R866_MAX_STEPS=19200 R866_LORA_R=32 R866_LORA_ALPHA=128 R866_BETA=0.3 R866_MAX_LEN=12288 R866_EPOCHS=4
export SRC=/root/mining_src/s4-h138-f43-tok-dpo-l2 OUT=/root/r866 DATA=/root/r866/dpo_duel_reason.jsonl LOG=/root/logs/r866_train.nohup SKIP_LOCAL_TKC=1
bash /root/mining_src/$EXP/start_r866.sh
echo "[r866-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) TRAIN launched pid=$(cat /root/logs/r866_train.pid) BASE=$BASE"
