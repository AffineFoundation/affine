#!/usr/bin/env bash
# p3883: R809 tammy Soft MidRank HiBeta SoftCtx Mega UltraLoLR on brave GPUs 4,5 (fill idle)
set -euo pipefail
exec >/root/logs/r809_lean_warm.log 2>&1
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=4,5 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[r809-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) start GPUs 4,5"
BASE=/root/hf/hub/models--tammyfritz--Affine-5hmwhnfbix-tammy2/snapshots/7e5fd5f87e82606c32d59c3d2350e3ddfe49c4b5
EXP=r809-tammy-offline-dpo-hialpha-midrank-lobeta-midctx-megasuperextrasteps-ep4-ultralolr
mkdir -p /root/r809 /root/logs /root/affine_data \
  /root/mining_src/s4-h138-f43-tok-dpo-l2 \
  /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r809/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  if [[ -s /root/mining_src/$EXP/dpo_duel_reason.jsonl ]]; then
    cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r798/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r798/dpo_duel_reason.jsonl "$DATA"
  else
    echo "FATAL missing MidCtx data"; exit 1
  fi
fi
n=$(wc -l <"$DATA"); echo "[r809-lean] data_lines=$n"; test "$n" -ge 200
cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
if [[ -f /root/logs/r809_train.pid ]]; then
  old=$(cat /root/logs/r809_train.pid 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "FATAL r809 already alive pid=$old"; exit 1
  fi
fi
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
  echo "[r809-lean] wait VRAM4+5 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo "FATAL GPUs 4,5 busy"; exit 1; }
rm -rf /root/r809/train; mkdir -p /root/r809/train; : >/root/logs/r809_train.nohup
chmod +x /root/mining_src/$EXP/*.sh
export BASE
export CUDA_VISIBLE_DEVICES=4,5 R809_LR=5e-7 R809_MAX_STEPS=19200 R809_LORA_R=32 R809_LORA_ALPHA=128 R809_BETA=0.02 R809_MAX_LEN=8192 R809_EPOCHS=4
export SRC=/root/mining_src/s4-h138-f43-tok-dpo-l2 OUT=/root/r809 DATA=/root/r809/dpo_duel_reason.jsonl LOG=/root/logs/r809_train.nohup SKIP_LOCAL_TKC=1
bash /root/mining_src/$EXP/start_r809.sh
echo "[r809-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) TRAIN launched pid=$(cat /root/logs/r809_train.pid) BASE=$BASE"
