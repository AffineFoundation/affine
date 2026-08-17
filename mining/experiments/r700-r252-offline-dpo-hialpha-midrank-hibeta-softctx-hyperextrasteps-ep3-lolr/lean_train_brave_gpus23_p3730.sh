#!/usr/bin/env bash
# p3730: R700 Soft MidRank HiBeta SoftCtx HyperExtra ep3×LoLR on idle brave GPUs 2,3 after R687 MERGE
set -euo pipefail
exec >/root/logs/r700_lean_warm.log 2>&1
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=2,3 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "[r700-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) start GPUs 2,3"
BASE=/root/hf/hub/models--unconst--Affine-5czsc2fc98-r252-merged/snapshots/b42d6245d77fe30885ea8a90387771e1bc465e0f
EXP=r700-r252-offline-dpo-hialpha-midrank-hibeta-softctx-hyperextrasteps-ep3-lolr
mkdir -p /root/r700 /root/logs /root/affine_data \
  /root/mining_src/s4-h138-f43-tok-dpo-l2 \
  /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r700/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  if [[ -s /root/mining_src/$EXP/dpo_duel_reason.jsonl ]]; then
    cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/mining_src/r687-r252-offline-dpo-hialpha-midrank-hibeta-softctx-ultraextrasteps-ep3-lolr/dpo_duel_reason.jsonl ]]; then
    cp -f /root/mining_src/r687-r252-offline-dpo-hialpha-midrank-hibeta-softctx-ultraextrasteps-ep3-lolr/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r687/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r687/dpo_duel_reason.jsonl "$DATA"
  else
    echo "FATAL missing Soft MidRank HiBeta SoftCtx data"; exit 1
  fi
fi
n=$(wc -l <"$DATA"); echo "[r700-lean] data_lines=$n"; test "$n" -ge 200
cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
if [[ -f /root/logs/r700_train.pid ]]; then
  old=$(cat /root/logs/r700_train.pid 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "FATAL r700 already alive pid=$old"; exit 1
  fi
fi
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2,3 | awk '{s+=$1} END{print s+0}')
  echo "[r700-lean] wait VRAM2+3 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2,3 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo "FATAL GPUs 2,3 busy"; exit 1; }
rm -rf /root/r700/train; mkdir -p /root/r700/train; : >/root/logs/r700_train.nohup
chmod +x /root/mining_src/$EXP/*.sh
export BASE
export CUDA_VISIBLE_DEVICES=2,3 R700_LR=1e-6 R700_MAX_STEPS=10800 R700_LORA_R=32 R700_LORA_ALPHA=128 R700_BETA=0.3 R700_MAX_LEN=12288 R700_EPOCHS=3
export SRC=/root/mining_src/s4-h138-f43-tok-dpo-l2 OUT=/root/r700 DATA=/root/r700/dpo_duel_reason.jsonl LOG=/root/logs/r700_train.nohup SKIP_LOCAL_TKC=1
bash /root/mining_src/$EXP/start_r700.sh
echo "[r700-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) TRAIN launched pid=$(cat /root/logs/r700_train.pid)"
