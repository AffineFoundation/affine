#!/usr/bin/env bash
# p3883: R810 tammy Soft HiRank HiBeta SoftCtx Mega UltraLoLR on brave GPUs 6,7 (fill idle)
set -euo pipefail
exec >/root/logs/r810_lean_warm.log 2>&1
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=6,7 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[r810-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) start GPUs 6,7"
BASE=/root/hf/hub/models--tammyfritz--Affine-5hmwhnfbix-tammy2/snapshots/7e5fd5f87e82606c32d59c3d2350e3ddfe49c4b5
EXP=r810-tammy-offline-dpo-hialpha-hirank-lobeta-softctx-megasuperextrasteps-ep4-ultralolr
mkdir -p /root/r810 /root/logs /root/affine_data \
  /root/mining_src/s4-h138-f43-tok-dpo-l2 \
  /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r810/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  if [[ -s /root/mining_src/$EXP/dpo_duel_reason.jsonl ]]; then
    cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r781/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r781/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r780/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r780/dpo_duel_reason.jsonl "$DATA"
  else
    echo "FATAL missing SoftCtx data"; exit 1
  fi
fi
n=$(wc -l <"$DATA"); echo "[r810-lean] data_lines=$n"; test "$n" -ge 200
cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
if [[ -f /root/logs/r810_train.pid ]]; then
  old=$(cat /root/logs/r810_train.pid 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "FATAL r810 already alive pid=$old"; exit 1
  fi
fi
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
  echo "[r810-lean] wait VRAM6+7 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo "FATAL GPUs 6,7 busy"; exit 1; }
rm -rf /root/r810/train; mkdir -p /root/r810/train; : >/root/logs/r810_train.nohup
chmod +x /root/mining_src/$EXP/*.sh
export BASE
export CUDA_VISIBLE_DEVICES=6,7 R810_LR=5e-7 R810_MAX_STEPS=19200 R810_LORA_R=64 R810_LORA_ALPHA=128 R810_BETA=0.02 R810_MAX_LEN=12288 R810_EPOCHS=4
export SRC=/root/mining_src/s4-h138-f43-tok-dpo-l2 OUT=/root/r810 DATA=/root/r810/dpo_duel_reason.jsonl LOG=/root/logs/r810_train.nohup SKIP_LOCAL_TKC=1
bash /root/mining_src/$EXP/start_r810.sh
echo "[r810-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) TRAIN launched pid=$(cat /root/logs/r810_train.pid) BASE=$BASE"
