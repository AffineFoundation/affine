#!/usr/bin/env bash
# p4117: R978 SoftCtx MidRank Hiβ Mega HiLR REFUTE m=-0.003060 ~-0.55× B✗0.297 → MidCtx MidRank Hiβ Mega HiLR
# ≠ SoftCtx Mega R978 / ≠ SoftCtx Ultra R964 / ≠ MidCtx Ultra UltraLoLR R955 / ≠ ShortCtx MidRank Hiβ Ultra HiLR R984 / ≠ SoftCtx MidRank Midβ Ultra R989 / ≠ Online / ≠ GRPO
set -euo pipefail
exec >/root/logs/r990_lean_warm.log 2>&1
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=6,7 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[r990-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) start GPUs 6,7 MidCtx MidRank Hiβ Mega HiLR"
BASE=/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/8e3f1695e058837ed80fec3238ff439fdc2d0f0e
EXP=r990-vera-offline-dpo-hialpha-midrank-hibeta-midctx-megasuperextrasteps-ep4-hilr
mkdir -p /root/r990 /root/logs /root/affine_data \
  /root/mining_src/s4-h138-f43-tok-dpo-l2 \
  /root/mining_src/s4-h1-sft \
  /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r990/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  if [[ -s /root/mining_src/$EXP/dpo_duel_reason.jsonl ]]; then cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r978/dpo_duel_reason.jsonl ]]; then cp -f /root/r978/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r964/dpo_duel_reason.jsonl ]]; then cp -f /root/r964/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r955/dpo_duel_reason.jsonl ]]; then cp -f /root/r955/dpo_duel_reason.jsonl "$DATA"
  else echo "FATAL missing Soft Mid Mid Soft data"; exit 1; fi
fi
n=$(wc -l <"$DATA"); echo "[r990-lean] data_lines=$n"; test "$n" -ge 200
cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
cp -f /root/mining_src/$EXP/merge_lora.py /root/mining_src/s4-h1-sft/merge_lora.py
if [[ -f /root/logs/r990_train.pid ]]; then
  old=$(cat /root/logs/r990_train.pid 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "FATAL r990 already alive pid=$old"; exit 1
  fi
fi
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
  echo "[r990-lean] wait VRAM6+7 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo "FATAL GPUs 6,7 busy"; exit 1; }
rm -rf /root/r990/train; mkdir -p /root/r990/train
rm -f /root/logs/r990_train.done /root/logs/r990_merge.done /root/logs/r990_merge_ready
: >/root/logs/r990_train.nohup
nohup python3 /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py \
  --base "$BASE" --data "$DATA" --out-dir /root/r990/train \
  --max-len 8192 --epochs 4 --lr 2e-6 \
  --lora-r 32 --lora-alpha 128 --beta 0.3 \
  --max-steps 19200 >/root/logs/r990_train.nohup 2>&1 &
echo $! | tee /root/logs/r990_train.pid >/root/r990/train.pid
python3 - <<'PY'
import json,time
from pathlib import Path
meta={
 "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
 "axis":"r990-vera-offline-dpo-hialpha-midrank-hibeta-midctx-megasuperextrasteps-ep4-hilr",
 "base":"vera6/affine-5g4yy75zuz-t6@8e3f1695",
 "lr":"2e-6","lora_r":32,"lora_alpha":128,"beta":0.3,"max_len":8192,"epochs":4,"max_steps":19200,
 "gpus":"6,7",
 "parent_signal":"R978 SoftCtx MidRank Hiβ Mega HiLR REFUTE m=-0.003060 ~-0.55× B✗0.297 (causality_fail) → MidCtx MidRank Hiβ Mega HiLR SoftCtx→MidCtx isolate; ≠ SoftCtx Mega R978 / ≠ SoftCtx Ultra R964 / ≠ MidCtx Ultra UltraLoLR R955 / ≠ ShortCtx MidRank Hiβ Ultra HiLR R984 / ≠ Online / ≠ GRPO",
 "decision_rule":"Stage-5 iff fresh v4 n80 margin>max(2*SE,0.002) AND thought>=80 AND B>=0.30 vs reign36"
}
Path("/root/affine_data/r990_train_launched.json").write_text(json.dumps(meta,indent=2)+"\n")
print(json.dumps(meta,indent=2))
PY
echo "[r990-lean] TRAIN launched pid=$(cat /root/logs/r990_train.pid) BASE=$BASE"
nohup bash /root/mining_src/$EXP/wait_r990_train_then_merge_p4117.sh >/dev/null 2>&1 &
echo $! >/root/logs/r990_wait_merge.pid
echo "[r990-lean] wait→merge armed pid=$(cat /root/logs/r990_wait_merge.pid)"
