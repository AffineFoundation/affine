#!/usr/bin/env bash
# p3807: R760 Short MidRank MidBeta MegaSuperExtra ep4 on R252 GPUs 6,7 after R751 REFUTE; leave R759 TRAIN on 4,5
set -euo pipefail
exec >/root/logs/r760_lean_warm.log 2>&1
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=6,7 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[r760-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) start GPUs 6,7"
BASE=/root/hf/hub/models--unconst--Affine-5czsc2fc98-r252-merged/snapshots/b42d6245d77fe30885ea8a90387771e1bc465e0f
EXP=r760-r252-offline-dpo-hialpha-midrank-midbeta-shortctx-megasuperextrasteps-ep4-lolr
mkdir -p /root/r760 /root/logs /root/affine_data \
  /root/mining_src/s4-h138-f43-tok-dpo-l2 \
  /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r760/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  if [[ -s /root/mining_src/$EXP/dpo_duel_reason.jsonl ]]; then
    cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r720/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r720/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r688/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r688/dpo_duel_reason.jsonl "$DATA"
  else
    echo "FATAL missing ShortCtx MidRank MidBeta data"; exit 1
  fi
fi
n=$(wc -l <"$DATA"); echo "[r760-lean] data_lines=$n"; test "$n" -ge 200
cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
if [[ -f /root/logs/r760_train.pid ]]; then
  old=$(cat /root/logs/r760_train.pid 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "FATAL r760 already alive pid=$old"; exit 1
  fi
fi
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
  echo "[r760-lean] wait VRAM6+7 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo "FATAL GPUs 6,7 busy"; exit 1; }
rm -rf /root/r760/train; mkdir -p /root/r760/train; : >/root/logs/r760_train.nohup
chmod +x /root/mining_src/$EXP/*.sh
export BASE
export CUDA_VISIBLE_DEVICES=6,7 R760_LR=1e-6 R760_MAX_STEPS=19200 R760_LORA_R=32 R760_LORA_ALPHA=128 R760_BETA=0.1 R760_MAX_LEN=6144 R760_EPOCHS=4
export SRC=/root/mining_src/s4-h138-f43-tok-dpo-l2 OUT=/root/r760 DATA=/root/r760/dpo_duel_reason.jsonl LOG=/root/logs/r760_train.nohup SKIP_LOCAL_TKC=1
bash /root/mining_src/$EXP/start_r760.sh
echo "[r760-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) TRAIN launched pid=$(cat /root/logs/r760_train.pid) BASE=$BASE"
