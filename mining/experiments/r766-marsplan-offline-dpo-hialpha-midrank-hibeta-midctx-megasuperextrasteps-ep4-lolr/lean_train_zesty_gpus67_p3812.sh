#!/usr/bin/env bash
# p3812: R766 marsplan MidCtx MidRank HiBeta MegaSuperExtra ep4 on zesty GPUs 6,7 after R754 REFUTE; leave R757 TRAIN on 4,5
set -euo pipefail
exec >/root/logs/r766_lean_warm.log 2>&1
set -a; source /root/mine.env; set +a
BASE=/root/hf/hub/models--marsplan0624--affine-5gedzafcvg-queen/snapshots/556d02a2adfa9bd42a02de3c766f98be7e44ca46
export BASE
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=6,7 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[r766-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) start GPUs 6,7"
EXP=r766-marsplan-offline-dpo-hialpha-midrank-hibeta-midctx-megasuperextrasteps-ep4-lolr
mkdir -p /root/r766 /root/logs /root/affine_data \
  /root/mining_src/s4-h138-f43-tok-dpo-l2 \
  /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r766/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  if [[ -s /root/mining_src/$EXP/dpo_duel_reason.jsonl ]]; then
    cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r723/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r723/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r733/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r733/dpo_duel_reason.jsonl "$DATA"
  else
    echo "FATAL missing MidCtx MidRank HiBeta data"; exit 1
  fi
fi
n=$(wc -l <"$DATA"); echo "[r766-lean] data_lines=$n"; test "$n" -ge 200
cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
if [[ -f /root/logs/r766_train.pid ]]; then
  old=$(cat /root/logs/r766_train.pid 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "FATAL r766 already alive pid=$old"; exit 1
  fi
fi
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
  echo "[r766-lean] wait VRAM6+7 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo "FATAL GPUs 6,7 busy"; exit 1; }
rm -rf /root/r766/train; mkdir -p /root/r766/train; : >/root/logs/r766_train.nohup
chmod +x /root/mining_src/$EXP/*.sh
export BASE
export CUDA_VISIBLE_DEVICES=6,7 R766_LR=1e-6 R766_MAX_STEPS=19200 R766_LORA_R=32 R766_LORA_ALPHA=128 R766_BETA=0.3 R766_MAX_LEN=8192 R766_EPOCHS=4
export SRC=/root/mining_src/s4-h138-f43-tok-dpo-l2 OUT=/root/r766 DATA=/root/r766/dpo_duel_reason.jsonl LOG=/root/logs/r766_train.nohup SKIP_LOCAL_TKC=1
bash /root/mining_src/$EXP/start_r766.sh
echo "[r766-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) TRAIN launched pid=$(cat /root/logs/r766_train.pid) BASE=$BASE"
