#!/usr/bin/env bash
# p3855: R798 tammy MidCtx MidRank MidBeta Mega UltraLoLR on crown GPUs 4,5 after R786 REFUTE; leave R799 on 6,7
set -euo pipefail
exec >/root/logs/r798_lean_warm.log 2>&1
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=4,5 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[r798-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) start GPUs 4,5"
BASE=/root/hf/hub/models--tammyfritz--Affine-5hmwhnfbix-tammy2/snapshots/7e5fd5f87e82606c32d59c3d2350e3ddfe49c4b5
EXP=r798-tammy-offline-dpo-hialpha-midrank-midbeta-midctx-megasuperextrasteps-ep4-ultralolr
mkdir -p /root/r798 /root/logs /root/affine_data \
  /root/mining_src/s4-h138-f43-tok-dpo-l2 \
  /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r798/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  if [[ -s /root/mining_src/$EXP/dpo_duel_reason.jsonl ]]; then
    cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r753/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r753/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r786/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r786/dpo_duel_reason.jsonl "$DATA"
  else
    echo "FATAL missing MidCtx MidBeta data"; exit 1
  fi
fi
n=$(wc -l <"$DATA"); echo "[r798-lean] data_lines=$n"; test "$n" -ge 200
cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
if [[ -f /root/logs/r798_train.pid ]]; then
  old=$(cat /root/logs/r798_train.pid 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "FATAL r798 already alive pid=$old"; exit 1
  fi
fi
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
  echo "[r798-lean] wait VRAM4+5 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo "FATAL GPUs 4,5 busy"; exit 1; }
rm -rf /root/r798/train; mkdir -p /root/r798/train; : >/root/logs/r798_train.nohup
chmod +x /root/mining_src/$EXP/*.sh
export BASE
export CUDA_VISIBLE_DEVICES=4,5 R798_LR=5e-7 R798_MAX_STEPS=19200 R798_LORA_R=32 R798_LORA_ALPHA=128 R798_BETA=0.1 R798_MAX_LEN=8192 R798_EPOCHS=4
export SRC=/root/mining_src/s4-h138-f43-tok-dpo-l2 OUT=/root/r798 DATA=/root/r798/dpo_duel_reason.jsonl LOG=/root/logs/r798_train.nohup SKIP_LOCAL_TKC=1
bash /root/mining_src/$EXP/start_r798.sh
echo "[r798-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) TRAIN launched pid=$(cat /root/logs/r798_train.pid) BASE=$BASE"
