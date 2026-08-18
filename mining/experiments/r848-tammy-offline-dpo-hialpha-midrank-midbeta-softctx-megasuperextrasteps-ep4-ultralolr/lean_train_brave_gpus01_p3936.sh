#!/usr/bin/env bash
# p3936: R848 on brave GPUs 0,1
set -euo pipefail
exec >/root/logs/r848_lean_warm.log 2>&1
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0,1 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[r848-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) start GPUs 0,1"
BASE=/root/hf/hub/models--tammyfritz--Affine-5hmwhnfbix-tammy2/snapshots/7e5fd5f87e82606c32d59c3d2350e3ddfe49c4b5
EXP=r848-tammy-offline-dpo-hialpha-midrank-midbeta-softctx-megasuperextrasteps-ep4-ultralolr
mkdir -p /root/r848 /root/logs /root/affine_data \
  /root/mining_src/s4-h138-f43-tok-dpo-l2 \
  /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r848/dpo_duel_reason.jsonl
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
n=$(wc -l <"$DATA"); echo "[r848-lean] data_lines=$n"; test "$n" -ge 200
cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
if [[ -f /root/logs/r848_train.pid ]]; then
  old=$(cat /root/logs/r848_train.pid 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "FATAL r848 already alive pid=$old"; exit 1
  fi
fi
G0=${CUDA_VISIBLE_DEVICES%%,*}; G1=${CUDA_VISIBLE_DEVICES##*,}
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G0,$G1 | awk '{s+=$1} END{print s+0}')
  echo "[r848-lean] wait VRAM used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G0,$G1 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo "FATAL GPUs 0,1 busy"; exit 1; }
rm -rf /root/r848/train; mkdir -p /root/r848/train; : >/root/logs/r848_train.nohup
chmod +x /root/mining_src/$EXP/*.sh
export BASE
export CUDA_VISIBLE_DEVICES=0,1 R848_LR=5e-7 R848_MAX_STEPS=19200 R848_LORA_R=32 R848_LORA_ALPHA=128 R848_BETA=0.1 R848_MAX_LEN=12288 R848_EPOCHS=4
export SRC=/root/mining_src/s4-h138-f43-tok-dpo-l2 OUT=/root/r848 DATA=/root/r848/dpo_duel_reason.jsonl LOG=/root/logs/r848_train.nohup SKIP_LOCAL_TKC=1
bash /root/mining_src/$EXP/start_r848.sh
echo "[r848-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) TRAIN launched pid=$(cat /root/logs/r848_train.pid) BASE=$BASE"
