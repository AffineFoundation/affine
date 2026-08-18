#!/usr/bin/env bash
# p3768: R728 marsplan Soft HiRank LoBeta MidCtx SuperExtra on lunar GPUs 6,7 after R712 REFUTE + R721 MERGE→chall on 4,5
set -euo pipefail
exec >/root/logs/r728_lean_warm.log 2>&1
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=6,7 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[r728-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) start GPUs 6,7"
# Pin BASE AFTER mine.env
BASE=/root/hf/hub/models--marsplan0624--affine-5gedzafcvg-queen/snapshots/556d02a2adfa9bd42a02de3c766f98be7e44ca46
EXP=r728-marsplan-offline-dpo-hialpha-hirank-lobeta-midctx-superextrasteps-ep3-lolr
mkdir -p /root/r728 /root/logs /root/affine_data \
  /root/mining_src/s4-h138-f43-tok-dpo-l2 \
  /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r728/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  if [[ -s /root/mining_src/$EXP/dpo_duel_reason.jsonl ]]; then
    cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r721/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r721/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/mining_src/r721-marsplan-offline-dpo-hialpha-midrank-lobeta-midctx-superextrasteps-ep3-lolr/dpo_duel_reason.jsonl ]]; then
    cp -f /root/mining_src/r721-marsplan-offline-dpo-hialpha-midrank-lobeta-midctx-superextrasteps-ep3-lolr/dpo_duel_reason.jsonl "$DATA"
  else
    echo "FATAL missing MidCtx data"; exit 1
  fi
fi
n=$(wc -l <"$DATA"); echo "[r728-lean] data_lines=$n"; test "$n" -ge 200
cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
if [[ -f /root/logs/r728_train.pid ]]; then
  old=$(cat /root/logs/r728_train.pid 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "FATAL r728 already alive pid=$old"; exit 1
  fi
fi
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
  echo "[r728-lean] wait VRAM6+7 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo "FATAL GPUs 6,7 busy"; exit 1; }
rm -rf /root/r728/train; mkdir -p /root/r728/train; : >/root/logs/r728_train.nohup
chmod +x /root/mining_src/$EXP/*.sh
export BASE
export CUDA_VISIBLE_DEVICES=6,7 R728_LR=1e-6 R728_MAX_STEPS=14400 R728_LORA_R=64 R728_LORA_ALPHA=128 R728_BETA=0.02 R728_MAX_LEN=8192 R728_EPOCHS=3
export SRC=/root/mining_src/s4-h138-f43-tok-dpo-l2 OUT=/root/r728 DATA=/root/r728/dpo_duel_reason.jsonl LOG=/root/logs/r728_train.nohup SKIP_LOCAL_TKC=1
bash /root/mining_src/$EXP/start_r728.sh
echo "[r728-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) TRAIN launched pid=$(cat /root/logs/r728_train.pid) BASE=$BASE"
