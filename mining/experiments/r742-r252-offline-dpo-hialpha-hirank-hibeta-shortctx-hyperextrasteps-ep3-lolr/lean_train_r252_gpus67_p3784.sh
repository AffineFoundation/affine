#!/usr/bin/env bash
# p3784: R742 on r252 GPUs 6,7 after R735 REFUTE
set -euo pipefail
exec >/root/logs/r742_lean_warm.log 2>&1
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=6,7 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[r742-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) start GPUs 6,7"
BASE=/root/hf/hub/models--unconst--Affine-5czsc2fc98-r252-merged/snapshots/b42d6245d77fe30885ea8a90387771e1bc465e0f
EXP=r742-r252-offline-dpo-hialpha-hirank-hibeta-shortctx-hyperextrasteps-ep3-lolr
mkdir -p /root/r742 /root/logs /root/affine_data \
  /root/mining_src/s4-h138-f43-tok-dpo-l2 \
  /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r742/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  if [[ -s /root/mining_src/$EXP/dpo_duel_reason.jsonl ]]; then
    cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r735/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r735/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r725/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r725/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r683/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r683/dpo_duel_reason.jsonl "$DATA"
  else
    echo "FATAL missing data"; exit 1
  fi
fi
n=$(wc -l <"$DATA"); echo "[r742-lean] data_lines=$n"; test "$n" -ge 200
cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
if [[ -f /root/logs/r742_train.pid ]]; then
  old=$(cat /root/logs/r742_train.pid 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "FATAL r742 already alive pid=$old"; exit 1
  fi
fi
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
  echo "[r742-lean] wait VRAM6+7 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo "FATAL GPUs 6,7 busy"; exit 1; }
rm -rf /root/r742/train; mkdir -p /root/r742/train; : >/root/logs/r742_train.nohup
chmod +x /root/mining_src/$EXP/*.sh
export BASE
export CUDA_VISIBLE_DEVICES=6,7 R742_LR=1e-6 R742_MAX_STEPS=10800 R742_LORA_R=64 R742_LORA_ALPHA=128 R742_BETA=0.3 R742_MAX_LEN=6144 R742_EPOCHS=3
export SRC=/root/mining_src/s4-h138-f43-tok-dpo-l2 OUT=/root/r742 DATA=/root/r742/dpo_duel_reason.jsonl LOG=/root/logs/r742_train.nohup SKIP_LOCAL_TKC=1
bash /root/mining_src/$EXP/start_r742.sh
echo "[r742-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) TRAIN launched pid=$(cat /root/logs/r742_train.pid) BASE=$BASE"
