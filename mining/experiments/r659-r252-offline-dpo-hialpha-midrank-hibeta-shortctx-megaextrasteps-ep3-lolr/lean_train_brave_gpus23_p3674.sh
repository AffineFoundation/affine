#!/usr/bin/env bash
# p3674: R659 Short MidRank HiBeta ShortCtx Mega ep3×LoLR on idle brave GPUs 2,3
set -euo pipefail
exec >/root/logs/r659_lean_warm.log 2>&1
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=2,3 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "[r659-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) start GPUs 2,3"
BASE=/root/hf/hub/models--unconst--Affine-5czsc2fc98-r252-merged/snapshots/b42d6245d77fe30885ea8a90387771e1bc465e0f
EXP=r659-r252-offline-dpo-hialpha-midrank-hibeta-shortctx-megaextrasteps-ep3-lolr
mkdir -p /root/r659 /root/logs /root/affine_data \
  /root/mining_src/s4-h138-f43-tok-dpo-l2 \
  /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r659/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  if [[ -s /root/mining_src/$EXP/dpo_duel_reason.jsonl ]]; then
    cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r627/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r627/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r632/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r632/dpo_duel_reason.jsonl "$DATA"
  else
    echo "FATAL missing Short MidRank HiBeta ShortCtx data"; exit 1
  fi
fi
n=$(wc -l <"$DATA"); echo "[r659-lean] data_lines=$n"; test "$n" -ge 200
cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
if [[ -f /root/logs/r659_train.pid ]]; then
  old=$(cat /root/logs/r659_train.pid 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "FATAL r659 already alive pid=$old"; exit 1
  fi
fi
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2,3 | awk '{s+=$1} END{print s+0}')
  echo "[r659-lean] wait VRAM2+3 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2,3 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo "FATAL GPUs 2,3 busy"; exit 1; }
rm -rf /root/r659/train; mkdir -p /root/r659/train; : >/root/logs/r659_train.nohup
chmod +x /root/mining_src/$EXP/*.sh
export CUDA_VISIBLE_DEVICES=2,3 R659_LR=1e-6 R659_MAX_STEPS=3600 R659_LORA_R=32 R659_LORA_ALPHA=128 R659_BETA=0.3 R659_MAX_LEN=6144 R659_EPOCHS=3
export BASE SRC=/root/mining_src/s4-h138-f43-tok-dpo-l2 OUT=/root/r659 DATA=/root/r659/dpo_duel_reason.jsonl LOG=/root/logs/r659_train.nohup SKIP_LOCAL_TKC=1
bash /root/mining_src/$EXP/start_r659.sh
echo "[r659-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) TRAIN launched pid=$(cat /root/logs/r659_train.pid)"
