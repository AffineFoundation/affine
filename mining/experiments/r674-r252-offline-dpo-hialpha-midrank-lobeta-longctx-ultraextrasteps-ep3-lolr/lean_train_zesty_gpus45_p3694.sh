#!/usr/bin/env bash
# p3694: R674 Long MidRank LoBeta UltraExtra ep3×LoLR on idle zesty GPUs 4,5
set -euo pipefail
exec >/root/logs/r674_lean_warm.log 2>&1
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=4,5 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "[r674-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) start GPUs 4,5"
BASE=/root/hf/hub/models--unconst--Affine-5czsc2fc98-r252-merged/snapshots/b42d6245d77fe30885ea8a90387771e1bc465e0f
EXP=r674-r252-offline-dpo-hialpha-midrank-lobeta-longctx-ultraextrasteps-ep3-lolr
mkdir -p /root/r674 /root/logs /root/affine_data \
  /root/mining_src/s4-h138-f43-tok-dpo-l2 \
  /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r674/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  if [[ -s /root/mining_src/$EXP/dpo_duel_reason.jsonl ]]; then
    cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r662/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r662/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/mining_src/r662-r252-offline-dpo-hialpha-midrank-lobeta-longctx-megaextrasteps-ep3-lolr/dpo_duel_reason.jsonl ]]; then
    cp -f /root/mining_src/r662-r252-offline-dpo-hialpha-midrank-lobeta-longctx-megaextrasteps-ep3-lolr/dpo_duel_reason.jsonl "$DATA"
  else
    echo "FATAL missing Long MidRank LoBeta data"; exit 1
  fi
fi
n=$(wc -l <"$DATA"); echo "[r674-lean] data_lines=$n"; test "$n" -ge 200
cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
if [[ -f /root/logs/r674_train.pid ]]; then
  old=$(cat /root/logs/r674_train.pid 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "FATAL r674 already alive pid=$old"; exit 1
  fi
fi
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
  echo "[r674-lean] wait VRAM4+5 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo "FATAL GPUs 4,5 busy"; exit 1; }
rm -rf /root/r674/train; mkdir -p /root/r674/train; : >/root/logs/r674_train.nohup
chmod +x /root/mining_src/$EXP/*.sh
# Pin BASE AFTER mine.env (p3689 lesson)
export BASE
export CUDA_VISIBLE_DEVICES=4,5 R674_LR=1e-6 R674_MAX_STEPS=7200 R674_LORA_R=32 R674_LORA_ALPHA=128 R674_BETA=0.02 R674_MAX_LEN=16384 R674_EPOCHS=3
export SRC=/root/mining_src/s4-h138-f43-tok-dpo-l2 OUT=/root/r674 DATA=/root/r674/dpo_duel_reason.jsonl LOG=/root/logs/r674_train.nohup SKIP_LOCAL_TKC=1
bash /root/mining_src/$EXP/start_r674.sh
echo "[r674-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) TRAIN launched pid=$(cat /root/logs/r674_train.pid)"
