#!/usr/bin/env bash
# p3708: R685 Long HiRank HiBeta UltraExtra ep3×LoLR on idle brave GPUs 0,1 after R669 MERGE
set -euo pipefail
exec >/root/logs/r685_lean_warm.log 2>&1
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0,1 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "[r685-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) start GPUs 0,1"
BASE=/root/hf/hub/models--unconst--Affine-5czsc2fc98-r252-merged/snapshots/b42d6245d77fe30885ea8a90387771e1bc465e0f
EXP=r685-r252-offline-dpo-hialpha-hirank-hibeta-longctx-ultraextrasteps-ep3-lolr
mkdir -p /root/r685 /root/logs /root/affine_data \
  /root/mining_src/s4-h138-f43-tok-dpo-l2 \
  /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r685/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  if [[ -s /root/mining_src/$EXP/dpo_duel_reason.jsonl ]]; then
    cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r669/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r669/dpo_duel_reason.jsonl "$DATA"
  else
    echo "FATAL missing Long HiRank HiBeta data"; exit 1
  fi
fi
n=$(wc -l <"$DATA"); echo "[r685-lean] data_lines=$n"; test "$n" -ge 200
cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
if [[ -f /root/logs/r685_train.pid ]]; then
  old=$(cat /root/logs/r685_train.pid 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "FATAL r685 already alive pid=$old"; exit 1
  fi
fi
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0,1 | awk '{s+=$1} END{print s+0}')
  echo "[r685-lean] wait VRAM0+1 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0,1 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo "FATAL GPUs 0,1 busy"; exit 1; }
rm -rf /root/r685/train; mkdir -p /root/r685/train; : >/root/logs/r685_train.nohup
chmod +x /root/mining_src/$EXP/*.sh
export BASE
export CUDA_VISIBLE_DEVICES=0,1 R685_LR=1e-6 R685_MAX_STEPS=7200 R685_LORA_R=64 R685_LORA_ALPHA=128 R685_BETA=0.3 R685_MAX_LEN=16384 R685_EPOCHS=3
export SRC=/root/mining_src/s4-h138-f43-tok-dpo-l2 OUT=/root/r685 DATA=/root/r685/dpo_duel_reason.jsonl LOG=/root/logs/r685_train.nohup SKIP_LOCAL_TKC=1
bash /root/mining_src/$EXP/start_r685.sh
echo "[r685-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) TRAIN launched pid=$(cat /root/logs/r685_train.pid)"
