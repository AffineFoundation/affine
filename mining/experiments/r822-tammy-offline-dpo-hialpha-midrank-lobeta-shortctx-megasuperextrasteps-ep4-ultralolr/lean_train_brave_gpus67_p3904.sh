#!/usr/bin/env bash
# p3904: R822 tammy Offline-DPO on brave GPUs 6,7 (fill idle)
set -euo pipefail
exec >/root/logs/r822_lean_warm.log 2>&1
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=6,7 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[r822-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) start GPUs 6,7"
BASE=/root/hf/hub/models--tammyfritz--Affine-5hmwhnfbix-tammy2/snapshots/7e5fd5f87e82606c32d59c3d2350e3ddfe49c4b5
EXP=r822-tammy-offline-dpo-hialpha-midrank-lobeta-shortctx-megasuperextrasteps-ep4-ultralolr
mkdir -p /root/r822 /root/logs /root/affine_data \
  /root/mining_src/s4-h138-f43-tok-dpo-l2 \
  /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r822/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  if [[ -s /root/mining_src/r822-tammy-offline-dpo-hialpha-midrank-lobeta-shortctx-megasuperextrasteps-ep4-ultralolr/dpo_duel_reason.jsonl ]]; then
    cp -f /root/mining_src/r822-tammy-offline-dpo-hialpha-midrank-lobeta-shortctx-megasuperextrasteps-ep4-ultralolr/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/mining_src/r653-r252-offline-dpo-hialpha-midrank-lobeta-shortctx-megaextrasteps-ep3-lolr/dpo_duel_reason.jsonl ]]; then
    cp -f /root/mining_src/r653-r252-offline-dpo-hialpha-midrank-lobeta-shortctx-megaextrasteps-ep3-lolr/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/mining_src/r581-r252-offline-dpo-hialpha-hirank-lobeta-shortctx-megaextrasteps/dpo_duel_reason.jsonl ]]; then
    cp -f /root/mining_src/r581-r252-offline-dpo-hialpha-hirank-lobeta-shortctx-megaextrasteps/dpo_duel_reason.jsonl "$DATA"
  else
    echo "FATAL missing ShortCtx data"; exit 1
  fi
fi
n=$(wc -l <"$DATA"); echo "[r822-lean] data_lines=$n"; test "$n" -ge 200
cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
if [[ -f /root/logs/r822_train.pid ]]; then
  old=$(cat /root/logs/r822_train.pid 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "FATAL r822 already alive pid=$old"; exit 1
  fi
fi
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
  echo "[r822-lean] wait VRAM6+7 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo "FATAL GPUs 6,7 busy"; exit 1; }
rm -rf /root/r822/train; mkdir -p /root/r822/train; : >/root/logs/r822_train.nohup
chmod +x /root/mining_src/$EXP/*.sh
export BASE
export CUDA_VISIBLE_DEVICES=6,7 R822_LR=5e-7 R822_MAX_STEPS=19200 R822_LORA_R=32 R822_LORA_ALPHA=128 R822_BETA=0.02 R822_MAX_LEN=6144 R822_EPOCHS=4
export SRC=/root/mining_src/s4-h138-f43-tok-dpo-l2 OUT=/root/r822 DATA=/root/r822/dpo_duel_reason.jsonl LOG=/root/logs/r822_train.nohup SKIP_LOCAL_TKC=1
bash /root/mining_src/$EXP/start_r822.sh
echo "[r822-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) TRAIN launched pid=$(cat /root/logs/r822_train.pid) BASE=$BASE"
