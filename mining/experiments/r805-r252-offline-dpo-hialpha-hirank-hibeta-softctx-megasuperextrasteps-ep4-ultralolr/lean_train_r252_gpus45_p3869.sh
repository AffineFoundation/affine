#!/usr/bin/env bash
# p3869: R805 r252 Soft HiRank HiBeta SoftCtx Mega UltraLoLR on R252 GPUs 4,5 after R793 REFUTE
# (R767 Soft Hi Hi Soft Mega LoLR near-miss ~0.47x → UltraLoLR sibling; ≠ R793 Soft Mid Mid Soft UltraLoLR; ≠ R784 tammy Soft Hi Hi Soft UltraLoLR; ≠ Online / ≠ GRPO)
set -euo pipefail
exec >/root/logs/r805_lean_warm.log 2>&1
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=4,5 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[r805-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) start GPUs 4,5"
BASE=/root/hf/hub/models--unconst--Affine-5czsc2fc98-r252-merged/snapshots/b42d6245d77fe30885ea8a90387771e1bc465e0f
EXP=r805-r252-offline-dpo-hialpha-hirank-hibeta-softctx-megasuperextrasteps-ep4-ultralolr
mkdir -p /root/r805 /root/logs /root/affine_data \
  /root/mining_src/s4-h138-f43-tok-dpo-l2 \
  /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r805/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  if [[ -s /root/mining_src/$EXP/dpo_duel_reason.jsonl ]]; then
    cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/mining_src/r667-r252-offline-dpo-hialpha-hirank-hibeta-softctx-megaextrasteps-ep3-lolr/dpo_duel_reason.jsonl ]]; then
    cp -f /root/mining_src/r667-r252-offline-dpo-hialpha-hirank-hibeta-softctx-megaextrasteps-ep3-lolr/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/mining_src/r602-r252-offline-dpo-hialpha-hirank-hibeta-softctx-megaextrasteps/dpo_duel_reason.jsonl ]]; then
    cp -f /root/mining_src/r602-r252-offline-dpo-hialpha-hirank-hibeta-softctx-megaextrasteps/dpo_duel_reason.jsonl "$DATA"
  else
    echo "FATAL missing Soft Hi Hi Soft data"; exit 1
  fi
fi
n=$(wc -l <"$DATA"); echo "[r805-lean] data_lines=$n"; test "$n" -ge 200
cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
if [[ -f /root/logs/r805_train.pid ]]; then
  old=$(cat /root/logs/r805_train.pid 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "FATAL r805 already alive pid=$old"; exit 1
  fi
fi
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
  echo "[r805-lean] wait VRAM4+5 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo "FATAL GPUs 4,5 busy"; exit 1; }
rm -rf /root/r805/train; mkdir -p /root/r805/train; : >/root/logs/r805_train.nohup
chmod +x /root/mining_src/$EXP/*.sh
export BASE
export CUDA_VISIBLE_DEVICES=4,5 R805_LR=5e-7 R805_MAX_STEPS=19200 R805_LORA_R=64 R805_LORA_ALPHA=128 R805_BETA=0.3 R805_MAX_LEN=12288 R805_EPOCHS=4
export SRC=/root/mining_src/s4-h138-f43-tok-dpo-l2 OUT=/root/r805 DATA=/root/r805/dpo_duel_reason.jsonl LOG=/root/logs/r805_train.nohup SKIP_LOCAL_TKC=1
bash /root/mining_src/$EXP/start_r805.sh
echo "[r805-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) TRAIN launched pid=$(cat /root/logs/r805_train.pid) BASE=$BASE"
