#!/usr/bin/env bash
# p4107: R972 SoftCtx LoRank Midβ Ultra HiLR REFUTE m=−0.011802 ~−0.89× → SoftCtx LoRank Midβ Mega HiLR
# ≠ Ultra R972 / ≠ SoftCtx LoRank Hiβ Ultra HiLR R967 / ≠ SoftCtx MidRank Midβ Mega HiLR R977 / ≠ SoftCtx LoRank Midβ Ultra UltraLoLR / ≠ Online / ≠ GRPO
set -euo pipefail
exec >/root/logs/r983_lean_warm.log 2>&1
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=1,3 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[r983-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) start GPUs 1,3 SoftCtx LoRank Midβ Mega HiLR"
BASE=/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/8e3f1695e058837ed80fec3238ff439fdc2d0f0e
EXP=r983-vera-offline-dpo-hialpha-lorank-midbeta-softctx-megasuperextrasteps-ep4-hilr
mkdir -p /root/r983 /root/logs /root/affine_data \
  /root/mining_src/s4-h138-f43-tok-dpo-l2 \
  /root/mining_src/s4-h1-sft \
  /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r983/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  if [[ -s /root/mining_src/$EXP/dpo_duel_reason.jsonl ]]; then cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r972/dpo_duel_reason.jsonl ]]; then cp -f /root/r972/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r970/dpo_duel_reason.jsonl ]]; then cp -f /root/r970/dpo_duel_reason.jsonl "$DATA"
  else echo "FATAL missing Soft Mid Mid Soft data"; exit 1; fi
fi
n=$(wc -l <"$DATA"); echo "[r983-lean] data_lines=$n"; test "$n" -ge 200
cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
cp -f /root/mining_src/$EXP/merge_lora.py /root/mining_src/s4-h1-sft/merge_lora.py
if [[ -f /root/logs/r983_train.pid ]]; then
  old=$(cat /root/logs/r983_train.pid 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "FATAL r983 already alive pid=$old"; exit 1
  fi
fi
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1,3 | awk '{s+=$1} END{print s+0}')
  echo "[r983-lean] wait VRAM1+3 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1,3 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo "FATAL GPUs 1,3 busy"; exit 1; }
rm -rf /root/r983/train; mkdir -p /root/r983/train
rm -f /root/logs/r983_train.done /root/logs/r983_merge.done /root/logs/r983_merge_ready
: >/root/logs/r983_train.nohup
nohup python3 /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py \
  --base "$BASE" --data "$DATA" --out-dir /root/r983/train \
  --max-len 12288 --epochs 4 --lr 2e-6 \
  --lora-r 16 --lora-alpha 128 --beta 0.1 \
  --max-steps 19200 >/root/logs/r983_train.nohup 2>&1 &
echo $! | tee /root/logs/r983_train.pid >/root/r983/train.pid
python3 - <<'PY'
import json,time
from pathlib import Path
meta={
 "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
 "axis":"r983-vera-offline-dpo-hialpha-lorank-midbeta-softctx-megasuperextrasteps-ep4-hilr",
 "base":"vera6/affine-5g4yy75zuz-t6@8e3f1695",
 "lr":"2e-6","lora_r":16,"lora_alpha":128,"beta":0.1,"max_len":12288,"epochs":4,"max_steps":19200,
 "gpus":"1,3",
 "parent_signal":"R972 SoftCtx LoRank Midβ Ultra HiLR REFUTE m=−0.011802 ~−0.89× → SoftCtx LoRank Midβ Mega HiLR isolate; ≠ Ultra R972 / ≠ SoftCtx LoRank Hiβ Ultra HiLR R967 / ≠ SoftCtx MidRank Midβ Mega HiLR R977 / ≠ Online / ≠ GRPO",
 "decision_rule":"Stage-5 iff fresh v4 n80 margin>max(2*SE,0.002) AND thought>=80 AND B>=0.30 vs reign36"
}
Path("/root/affine_data/r983_train_launched.json").write_text(json.dumps(meta,indent=2)+"\n")
print(json.dumps(meta,indent=2))
PY
echo "[r983-lean] TRAIN launched pid=$(cat /root/logs/r983_train.pid)"
nohup bash /root/mining_src/$EXP/wait_r983_train_then_merge_p4107.sh >/dev/null 2>&1 &
echo $! >/root/logs/r983_wait_merge.pid
