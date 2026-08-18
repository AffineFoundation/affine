#!/usr/bin/env bash
# p3950: R860 cryptoDev23 SoftCtx HiRank MidBeta UltraLoLR after R830 REFUTE ~-0.21×
set -euo pipefail
exec >/root/logs/r860_lean_warm.log 2>&1
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=6,7 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[r860-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) start GPUs 6,7"
BASE=/root/hf/hub/models--cryptoDev23--Affine-5Dku3dYp9j-hk8161/snapshots/55b7ffe003d078a8a131673f677b2584548a502e
EXP=r860-cryptodev23-offline-dpo-hialpha-hirank-midbeta-softctx-megasuperextrasteps-ep4-ultralolr
mkdir -p /root/r860 /root/logs /root/affine_data \
  /root/mining_src/s4-h138-f43-tok-dpo-l2 \
  /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r860/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  if [[ -s /root/r808/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r808/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r843/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r843/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r830/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r830/dpo_duel_reason.jsonl "$DATA"
  else
    echo "FATAL missing Soft Mid Mid Soft SoftCtx DPO data"; exit 1
  fi
fi
n=$(wc -l <"$DATA"); echo "[r860-lean] data_lines=$n"; test "$n" -ge 200
cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
if [[ -f /root/logs/r860_train.pid ]]; then
  old=$(cat /root/logs/r860_train.pid 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "FATAL r860 already alive pid=$old"; exit 1
  fi
fi
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
  echo "[r860-lean] wait VRAM6+7 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo "FATAL GPUs 6,7 busy"; exit 1; }
rm -rf /root/r860/train; mkdir -p /root/r860/train; : >/root/logs/r860_train.nohup
chmod +x /root/mining_src/$EXP/*.sh
export BASE
export CUDA_VISIBLE_DEVICES=6,7 R860_LR=5e-7 R860_MAX_STEPS=19200 R860_LORA_R=64 R860_LORA_ALPHA=128 R860_BETA=0.1 R860_MAX_LEN=12288 R860_EPOCHS=4
export SRC=/root/mining_src/s4-h138-f43-tok-dpo-l2 OUT=/root/r860 DATA=/root/r860/dpo_duel_reason.jsonl LOG=/root/logs/r860_train.nohup SKIP_LOCAL_TKC=1
bash /root/mining_src/$EXP/start_r860.sh
echo "[r860-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) TRAIN launched pid=$(cat /root/logs/r860_train.pid) BASE=$BASE"
