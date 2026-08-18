#!/usr/bin/env bash
# p3841: R787 marsplan SoftCtx MidRank HiBeta MegaSuperExtra ep4 on zesty GPUs 6,7 after R777 REFUTE; leave R785 TRAIN on 4,5
set -euo pipefail
exec >/root/logs/r787_lean_warm.log 2>&1
set -a; source /root/mine.env; set +a
BASE=/root/hf/hub/models--marsplan0624--affine-5gedzafcvg-queen/snapshots/556d02a2adfa9bd42a02de3c766f98be7e44ca46
export BASE
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=6,7 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[r787-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) start GPUs 6,7"
EXP=r787-marsplan-offline-dpo-hialpha-midrank-hibeta-softctx-megasuperextrasteps-ep4-ultralolr
mkdir -p /root/r787 /root/logs /root/affine_data \
  /root/mining_src/s4-h138-f43-tok-dpo-l2 \
  /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r787/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  if [[ -s /root/mining_src/$EXP/dpo_duel_reason.jsonl ]]; then
    cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r777/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r777/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r777/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r777/dpo_duel_reason.jsonl "$DATA"
  else
    echo "FATAL missing SoftCtx MidRank HiBeta data"; exit 1
  fi
fi
n=$(wc -l <"$DATA"); echo "[r787-lean] data_lines=$n"; test "$n" -ge 200
cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
if [[ -f /root/logs/r787_train.pid ]]; then
  old=$(cat /root/logs/r787_train.pid 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "FATAL r787 already alive pid=$old"; exit 1
  fi
fi
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
  echo "[r787-lean] wait VRAM6+7 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo "FATAL GPUs 6,7 busy"; exit 1; }
rm -rf /root/r787/train; mkdir -p /root/r787/train; : >/root/logs/r787_train.nohup
chmod +x /root/mining_src/$EXP/*.sh
export BASE
export CUDA_VISIBLE_DEVICES=6,7 R787_LR=5e-7 R787_MAX_STEPS=19200 R787_LORA_R=32 R787_LORA_ALPHA=128 R787_BETA=0.3 R787_MAX_LEN=12288 R787_EPOCHS=4
export SRC=/root/mining_src/s4-h138-f43-tok-dpo-l2 OUT=/root/r787 DATA=/root/r787/dpo_duel_reason.jsonl LOG=/root/logs/r787_train.nohup SKIP_LOCAL_TKC=1
bash /root/mining_src/$EXP/start_r787.sh
echo "[r787-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) TRAIN launched pid=$(cat /root/logs/r787_train.pid) BASE=$BASE"
