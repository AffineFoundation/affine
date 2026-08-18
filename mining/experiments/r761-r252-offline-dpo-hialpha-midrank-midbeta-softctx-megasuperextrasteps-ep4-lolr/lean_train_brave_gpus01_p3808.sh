#!/usr/bin/env bash
# p3808: R761 Soft MidRank MidBeta SoftCtx MegaSuperExtra ep4 on brave GPUs 0,1
set -euo pipefail
exec >/root/logs/r761_lean_warm.log 2>&1
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0,1 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[r761-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) start GPUs 0,1"
BASE=/root/hf/hub/models--unconst--Affine-5czsc2fc98-r252-merged/snapshots/b42d6245d77fe30885ea8a90387771e1bc465e0f
EXP=r761-r252-offline-dpo-hialpha-midrank-midbeta-softctx-megasuperextrasteps-ep4-lolr
mkdir -p /root/r761 /root/logs /root/affine_data /root/mining_src/s4-h138-f43-tok-dpo-l2 /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r761/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  if [[ -s /root/mining_src/$EXP/dpo_duel_reason.jsonl ]]; then
    cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/mining_src/r630-r252-offline-dpo-hialpha-midrank-midbeta-softctx-megaextrasteps-ep2-lolr/dpo_duel_reason.jsonl ]]; then
    cp -f /root/mining_src/r630-r252-offline-dpo-hialpha-midrank-midbeta-softctx-megaextrasteps-ep2-lolr/dpo_duel_reason.jsonl "$DATA"
  else
    echo "FATAL missing SoftCtx MidRank MidBeta data"; exit 1
  fi
fi
n=$(wc -l <"$DATA"); echo "[r761-lean] data_lines=$n"; test "$n" -ge 200
cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
if [[ -f /root/logs/r761_train.pid ]]; then
  old=$(cat /root/logs/r761_train.pid 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "FATAL r761 already alive pid=$old"; exit 1
  fi
fi
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0,1 | awk '{s+=$1} END{print s+0}')
  echo "[r761-lean] wait VRAM0+1 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0,1 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo "FATAL GPUs 0,1 busy"; exit 1; }
rm -rf /root/r761/train; mkdir -p /root/r761/train; : >/root/logs/r761_train.nohup
chmod +x /root/mining_src/$EXP/*.sh
export BASE
export CUDA_VISIBLE_DEVICES=0,1 R761_LR=1e-6 R761_MAX_STEPS=19200 R761_LORA_R=32 R761_LORA_ALPHA=128 R761_BETA=0.1 R761_MAX_LEN=12288 R761_EPOCHS=4
export SRC=/root/mining_src/s4-h138-f43-tok-dpo-l2 OUT=/root/r761 DATA=/root/r761/dpo_duel_reason.jsonl LOG=/root/logs/r761_train.nohup SKIP_LOCAL_TKC=1
bash /root/mining_src/$EXP/start_r761.sh
echo "[r761-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) TRAIN launched pid=$(cat /root/logs/r761_train.pid) BASE=$BASE"
