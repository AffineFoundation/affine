#!/usr/bin/env bash
# p3784: R745 on golden GPUs 4,5 after R729 REFUTE
set -euo pipefail
exec >/root/logs/r745_lean_warm.log 2>&1
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=4,5 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[r745-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) start GPUs 4,5"
BASE=/root/hf/hub/models--unconst--Affine-5czsc2fc98-r252-merged/snapshots/b42d6245d77fe30885ea8a90387771e1bc465e0f
EXP=r745-r252-offline-dpo-hialpha-hirank-hibeta-midctx-superextrasteps-ep3-lolr
mkdir -p /root/r745 /root/logs /root/affine_data \
  /root/mining_src/s4-h138-f43-tok-dpo-l2 \
  /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r745/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  if [[ -s /root/mining_src/$EXP/dpo_duel_reason.jsonl ]]; then
    cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r729/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r729/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r719/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r719/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r684/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r684/dpo_duel_reason.jsonl "$DATA"
  else
    echo "FATAL missing data"; exit 1
  fi
fi
n=$(wc -l <"$DATA"); echo "[r745-lean] data_lines=$n"; test "$n" -ge 200
cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
if [[ -f /root/logs/r745_train.pid ]]; then
  old=$(cat /root/logs/r745_train.pid 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "FATAL r745 already alive pid=$old"; exit 1
  fi
fi
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
  echo "[r745-lean] wait VRAM4+5 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo "FATAL GPUs 4,5 busy"; exit 1; }
rm -rf /root/r745/train; mkdir -p /root/r745/train; : >/root/logs/r745_train.nohup
chmod +x /root/mining_src/$EXP/*.sh
export BASE
export CUDA_VISIBLE_DEVICES=4,5 R745_LR=1e-6 R745_MAX_STEPS=14400 R745_LORA_R=64 R745_LORA_ALPHA=128 R745_BETA=0.3 R745_MAX_LEN=8192 R745_EPOCHS=3
export SRC=/root/mining_src/s4-h138-f43-tok-dpo-l2 OUT=/root/r745 DATA=/root/r745/dpo_duel_reason.jsonl LOG=/root/logs/r745_train.nohup SKIP_LOCAL_TKC=1
bash /root/mining_src/$EXP/start_r745.sh
echo "[r745-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) TRAIN launched pid=$(cat /root/logs/r745_train.pid) BASE=$BASE"
