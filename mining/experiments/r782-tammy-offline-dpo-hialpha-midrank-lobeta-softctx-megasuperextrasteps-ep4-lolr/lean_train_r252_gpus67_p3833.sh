#!/usr/bin/env bash
# p3833: R782 tammy Soft MidRank LoBeta SoftCtx Mega on R252 GPUs 6,7 after R770 REFUTE
set -euo pipefail
exec >/root/logs/r782_lean_warm.log 2>&1
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=6,7 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[r782-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) start GPUs 6,7"
BASE=/root/hf/hub/models--tammyfritz--Affine-5hmwhnfbix-tammy2/snapshots/7e5fd5f87e82606c32d59c3d2350e3ddfe49c4b5
EXP=r782-tammy-offline-dpo-hialpha-midrank-lobeta-softctx-megasuperextrasteps-ep4-lolr
mkdir -p /root/r782 /root/logs /root/affine_data \
  /root/mining_src/s4-h138-f43-tok-dpo-l2 \
  /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r782/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  if [[ -s /root/mining_src/$EXP/dpo_duel_reason.jsonl ]]; then
    cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r775/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r775/dpo_duel_reason.jsonl "$DATA"
  else
    echo "FATAL missing SoftCtx data"; exit 1
  fi
fi
n=$(wc -l <"$DATA"); echo "[r782-lean] data_lines=$n"; test "$n" -ge 200
cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
if [[ -f /root/logs/r782_train.pid ]]; then
  old=$(cat /root/logs/r782_train.pid 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "FATAL r782 already alive pid=$old"; exit 1
  fi
fi
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
  echo "[r782-lean] wait VRAM6+7 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo "FATAL GPUs 6,7 busy"; exit 1; }
rm -rf /root/r782/train; mkdir -p /root/r782/train; : >/root/logs/r782_train.nohup
chmod +x /root/mining_src/$EXP/*.sh
export BASE
export CUDA_VISIBLE_DEVICES=6,7 R782_LR=1e-6 R782_MAX_STEPS=19200 R782_LORA_R=32 R782_LORA_ALPHA=128 R782_BETA=0.02 R782_MAX_LEN=12288 R782_EPOCHS=4
export SRC=/root/mining_src/s4-h138-f43-tok-dpo-l2 OUT=/root/r782 DATA=/root/r782/dpo_duel_reason.jsonl LOG=/root/logs/r782_train.nohup SKIP_LOCAL_TKC=1
bash /root/mining_src/$EXP/start_r782.sh
echo "[r782-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) TRAIN launched pid=$(cat /root/logs/r782_train.pid) BASE=$BASE"
