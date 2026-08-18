#!/usr/bin/env bash
# p3936: R851 on brave GPUs 6,7
set -euo pipefail
exec >/root/logs/r851_lean_warm.log 2>&1
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=6,7 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[r851-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) start GPUs 6,7"
BASE=/root/hf/hub/models--tammyfritz--Affine-5hmwhnfbix-tammy2/snapshots/7e5fd5f87e82606c32d59c3d2350e3ddfe49c4b5
EXP=r851-tammy-offline-dpo-hialpha-hirank-lobeta-softctx-megasuperextrasteps-ep4-ultralolr
mkdir -p /root/r851 /root/logs /root/affine_data \
  /root/mining_src/s4-h138-f43-tok-dpo-l2 \
  /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r851/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  if [[ -s /root/mining_src/$EXP/dpo_duel_reason.jsonl ]]; then
    cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r801/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r801/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r800/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r800/dpo_duel_reason.jsonl "$DATA"
  else
    echo "FATAL missing Soft Mid Mid Soft data"; exit 1
  fi
fi
n=$(wc -l <"$DATA"); echo "[r851-lean] data_lines=$n"; test "$n" -ge 200
cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
if [[ -f /root/logs/r851_train.pid ]]; then
  old=$(cat /root/logs/r851_train.pid 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "FATAL r851 already alive pid=$old"; exit 1
  fi
fi
G0=${CUDA_VISIBLE_DEVICES%%,*}; G1=${CUDA_VISIBLE_DEVICES##*,}
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G0,$G1 | awk '{s+=$1} END{print s+0}')
  echo "[r851-lean] wait VRAM used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G0,$G1 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo "FATAL GPUs 6,7 busy"; exit 1; }
rm -rf /root/r851/train; mkdir -p /root/r851/train; : >/root/logs/r851_train.nohup
chmod +x /root/mining_src/$EXP/*.sh
export BASE
export CUDA_VISIBLE_DEVICES=6,7 R851_LR=5e-7 R851_MAX_STEPS=19200 R851_LORA_R=64 R851_LORA_ALPHA=128 R851_BETA=0.02 R851_MAX_LEN=12288 R851_EPOCHS=4
export SRC=/root/mining_src/s4-h138-f43-tok-dpo-l2 OUT=/root/r851 DATA=/root/r851/dpo_duel_reason.jsonl LOG=/root/logs/r851_train.nohup SKIP_LOCAL_TKC=1
bash /root/mining_src/$EXP/start_r851.sh
echo "[r851-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) TRAIN launched pid=$(cat /root/logs/r851_train.pid) BASE=$BASE"
