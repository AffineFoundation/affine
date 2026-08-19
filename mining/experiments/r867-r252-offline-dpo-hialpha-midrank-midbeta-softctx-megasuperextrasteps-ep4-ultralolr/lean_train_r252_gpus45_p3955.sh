#!/usr/bin/env bash
set -euo pipefail
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH" PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=4,5 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[r867-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) GPUs 4,5"
BASE=/root/hf/hub/models--unconst--Affine-5czsc2fc98-r252-merged/snapshots/b42d6245d77fe30885ea8a90387771e1bc465e0f
EXP=r867-r252-offline-dpo-hialpha-midrank-midbeta-softctx-megasuperextrasteps-ep4-ultralolr
mkdir -p /root/r867 /root/logs /root/affine_data /root/mining_src/s4-h138-f43-tok-dpo-l2 /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r867/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  if [[ -s /root/r844/dpo_duel_reason.jsonl ]]; then cp -f /root/r844/dpo_duel_reason.jsonl "$DATA"
  else echo FATAL missing Soft Mid Mid Soft data; exit 1; fi
fi
n=$(wc -l <"$DATA"); test "$n" -ge 200
cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
  echo "[r867-lean] VRAM=$used iter=$i"; [[ "$used" -lt 8192 ]] && break; sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo FATAL busy; exit 1; }
rm -rf /root/r867/train; mkdir -p /root/r867/train; : >/root/logs/r867_train.nohup
chmod +x /root/mining_src/$EXP/*.sh
export BASE CUDA_VISIBLE_DEVICES=4,5 R867_LR=5e-7 R867_MAX_STEPS=19200 R867_LORA_R=32 R867_LORA_ALPHA=128 R867_BETA=0.1 R867_MAX_LEN=12288 R867_EPOCHS=4
export SRC=/root/mining_src/s4-h138-f43-tok-dpo-l2 OUT=/root/r867 DATA=/root/r867/dpo_duel_reason.jsonl LOG=/root/logs/r867_train.nohup SKIP_LOCAL_TKC=1
bash /root/mining_src/$EXP/start_r867.sh
echo "[r867-lean] TRAIN pid=$(cat /root/logs/r867_train.pid)"
