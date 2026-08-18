#!/usr/bin/env bash
# p3860: R802 r252 Soft MidRank LoBeta SoftCtx Mega UltraLoLR on R252 GPUs 6,7 after R793 REFUTE
# (R767 Soft Mid Lo Soft Mega LoLR near-miss ~0.47x → UltraLoLR sibling; ≠ R793 Soft Mid Mid Soft UltraLoLR; ≠ R784 tammy Soft Mid Lo Soft UltraLoLR; ≠ Online / ≠ GRPO)
set -euo pipefail
exec >/root/logs/r802_lean_warm.log 2>&1
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=6,7 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[r802-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) start GPUs 6,7"
BASE=/root/hf/hub/models--unconst--Affine-5czsc2fc98-r252-merged/snapshots/b42d6245d77fe30885ea8a90387771e1bc465e0f
EXP=r802-r252-offline-dpo-hialpha-midrank-lobeta-softctx-megasuperextrasteps-ep4-ultralolr
mkdir -p /root/r802 /root/logs /root/affine_data \
  /root/mining_src/s4-h138-f43-tok-dpo-l2 \
  /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r802/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  if [[ -s /root/mining_src/$EXP/dpo_duel_reason.jsonl ]]; then
    cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r782/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r782/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r767/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r767/dpo_duel_reason.jsonl "$DATA"
  else
    echo "FATAL missing Soft Mid Lo Soft data"; exit 1
  fi
fi
n=$(wc -l <"$DATA"); echo "[r802-lean] data_lines=$n"; test "$n" -ge 200
cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
if [[ -f /root/logs/r802_train.pid ]]; then
  old=$(cat /root/logs/r802_train.pid 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "FATAL r802 already alive pid=$old"; exit 1
  fi
fi
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
  echo "[r802-lean] wait VRAM6+7 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo "FATAL GPUs 6,7 busy"; exit 1; }
rm -rf /root/r802/train; mkdir -p /root/r802/train; : >/root/logs/r802_train.nohup
chmod +x /root/mining_src/$EXP/*.sh
export BASE
export CUDA_VISIBLE_DEVICES=6,7 R802_LR=5e-7 R802_MAX_STEPS=19200 R802_LORA_R=32 R802_LORA_ALPHA=128 R802_BETA=0.02 R802_MAX_LEN=12288 R802_EPOCHS=4
export SRC=/root/mining_src/s4-h138-f43-tok-dpo-l2 OUT=/root/r802 DATA=/root/r802/dpo_duel_reason.jsonl LOG=/root/logs/r802_train.nohup SKIP_LOCAL_TKC=1
bash /root/mining_src/$EXP/start_r802.sh
echo "[r802-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) TRAIN launched pid=$(cat /root/logs/r802_train.pid) BASE=$BASE"
