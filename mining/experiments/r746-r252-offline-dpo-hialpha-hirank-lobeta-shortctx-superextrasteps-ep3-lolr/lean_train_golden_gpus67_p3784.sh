#!/usr/bin/env bash
set -euo pipefail
exec >/root/logs/r746_lean_warm.log 2>&1
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=6,7 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[r746-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) start GPUs 6,7"
BASE=/root/hf/hub/models--unconst--Affine-5czsc2fc98-r252-merged/snapshots/b42d6245d77fe30885ea8a90387771e1bc465e0f
EXP=r746-r252-offline-dpo-hialpha-hirank-lobeta-shortctx-superextrasteps-ep3-lolr
mkdir -p /root/r746 /root/logs /root/affine_data /root/mining_src/s4-h138-f43-tok-dpo-l2 /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r746/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  if [[ -s /root/mining_src/$EXP/dpo_duel_reason.jsonl ]]; then cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r730/dpo_duel_reason.jsonl ]]; then cp -f /root/r730/dpo_duel_reason.jsonl "$DATA"
  else echo FATAL; exit 1; fi
fi
n=$(wc -l <"$DATA"); echo "[r746-lean] data_lines=$n"; test "$n" -ge 200
cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
  echo "[r746-lean] wait used=$used"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo FATAL busy; exit 1; }
rm -rf /root/r746/train; mkdir -p /root/r746/train; : >/root/logs/r746_train.nohup
chmod +x /root/mining_src/$EXP/*.sh
export BASE CUDA_VISIBLE_DEVICES=6,7 R746_LR=1e-6 R746_MAX_STEPS=14400 R746_LORA_R=64 R746_LORA_ALPHA=128 R746_BETA=0.02 R746_MAX_LEN=6144 R746_EPOCHS=3
export SRC=/root/mining_src/s4-h138-f43-tok-dpo-l2 OUT=/root/r746 DATA=/root/r746/dpo_duel_reason.jsonl LOG=/root/logs/r746_train.nohup SKIP_LOCAL_TKC=1
bash /root/mining_src/$EXP/start_r746.sh
echo "[r746-lean] TRAIN launched pid=$(cat /root/logs/r746_train.pid)"
