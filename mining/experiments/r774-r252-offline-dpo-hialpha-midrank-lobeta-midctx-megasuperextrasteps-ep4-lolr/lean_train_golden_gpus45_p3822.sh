#!/usr/bin/env bash
# p3822: R774 r252 MidCtx MidRank LoBeta MegaSuperExtra ep4 on golden GPUs 4,5 after R755 REFUTE; leave R773 TRAIN on 4,5
set -euo pipefail
exec >/root/logs/r774_lean_warm.log 2>&1
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=4,5 R774_LR=1e-6 R774_MAX_STEPS=19200 R774_LORA_R=32 R774_LORA_ALPHA=128 R774_BETA=0.02 R774_MAX_LEN=8192 R774_EPOCHS=4
echo "[r774-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) start GPUs 4,5"
BASE=/root/hf/hub/models--unconst--Affine-5czsc2fc98-r252-merged/snapshots/b42d6245d77fe30885ea8a90387771e1bc465e0f
EXP=r774-r252-offline-dpo-hialpha-midrank-lobeta-midctx-megasuperextrasteps-ep4-lolr
mkdir -p /root/r774 /root/logs /root/affine_data \
  /root/mining_src/s4-h138-f43-tok-dpo-l2 \
  /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r774/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  if [[ -s /root/mining_src/$EXP/dpo_duel_reason.jsonl ]]; then
    cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r736/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r736/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r745/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r745/dpo_duel_reason.jsonl "$DATA"
  else
    echo "FATAL missing MidCtx LoBeta data"; exit 1
  fi
fi
n=$(wc -l <"$DATA"); echo "[r774-lean] data_lines=$n"; test "$n" -ge 200
cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
if [[ -f /root/logs/r774_train.pid ]]; then
  old=$(cat /root/logs/r774_train.pid 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "FATAL r774 already alive pid=$old"; exit 1
  fi
fi
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
  echo "[r774-lean] wait VRAM6+7 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo "FATAL GPUs 4,5 busy"; exit 1; }
rm -rf /root/r774/train; mkdir -p /root/r774/train; : >/root/logs/r774_train.nohup
chmod +x /root/mining_src/$EXP/*.sh
export BASE
export CUDA_VISIBLE_DEVICES=4,5 R774_LR=1e-6 R774_MAX_STEPS=19200 R774_LORA_R=32 R774_LORA_ALPHA=128 R774_BETA=0.02 R774_MAX_LEN=8192 R774_EPOCHS=4
export SRC=/root/mining_src/s4-h138-f43-tok-dpo-l2 OUT=/root/r774 DATA=/root/r774/dpo_duel_reason.jsonl LOG=/root/logs/r774_train.nohup SKIP_LOCAL_TKC=1
bash /root/mining_src/$EXP/start_r774.sh
echo "[r774-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) TRAIN launched pid=$(cat /root/logs/r774_train.pid) BASE=$BASE"
