#!/usr/bin/env bash
# p3804: R757 marsplan SoftCtx HiRank MidBeta MegaSuperExtra ep4 on zesty GPUs 4,5 after R748 REFUTE; leave R754 TRAIN on 6,7
set -euo pipefail
exec >/root/logs/r757_lean_warm.log 2>&1
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=4,5 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[r757-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) start GPUs 4,5"
BASE=/root/hf/hub/models--marsplan0624--affine-5gedzafcvg-queen/snapshots/556d02a2adfa9bd42a02de3c766f98be7e44ca46
EXP=r757-marsplan-offline-dpo-hialpha-hirank-midbeta-softctx-megasuperextrasteps-ep4-lolr
mkdir -p /root/r757 /root/logs /root/affine_data \
  /root/mining_src/s4-h138-f43-tok-dpo-l2 \
  /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r757/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  if [[ -s /root/mining_src/$EXP/dpo_duel_reason.jsonl ]]; then
    cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r748/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r748/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r708/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r708/dpo_duel_reason.jsonl "$DATA"
  else
    echo "FATAL missing SoftCtx MidBeta data"; exit 1
  fi
fi
n=$(wc -l <"$DATA"); echo "[r757-lean] data_lines=$n"; test "$n" -ge 200
cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
if [[ -f /root/logs/r757_train.pid ]]; then
  old=$(cat /root/logs/r757_train.pid 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "FATAL r757 already alive pid=$old"; exit 1
  fi
fi
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
  echo "[r757-lean] wait VRAM4+5 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo "FATAL GPUs 4,5 busy"; exit 1; }
rm -rf /root/r757/train; mkdir -p /root/r757/train; : >/root/logs/r757_train.nohup
chmod +x /root/mining_src/$EXP/*.sh
export BASE
export CUDA_VISIBLE_DEVICES=4,5 R757_LR=1e-6 R757_MAX_STEPS=19200 R757_LORA_R=64 R757_LORA_ALPHA=128 R757_BETA=0.1 R757_MAX_LEN=12288 R757_EPOCHS=4
export SRC=/root/mining_src/s4-h138-f43-tok-dpo-l2 OUT=/root/r757 DATA=/root/r757/dpo_duel_reason.jsonl LOG=/root/logs/r757_train.nohup SKIP_LOCAL_TKC=1
bash /root/mining_src/$EXP/start_r757.sh
echo "[r757-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) TRAIN launched pid=$(cat /root/logs/r757_train.pid) BASE=$BASE"
