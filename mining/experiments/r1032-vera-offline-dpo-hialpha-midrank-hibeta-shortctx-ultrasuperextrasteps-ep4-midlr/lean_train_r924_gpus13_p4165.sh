#!/usr/bin/env bash
# p4165: R1015 ShortCtx MidRank Hiβ Mega MidLR REFUTE m=-0.002353 ~-0.94× → ShortCtx MidRank Hiβ Ultra MidLR
# Isolates Mega→Ultra (max_steps 19200→28800) on the R1015 ShortCtx MidRank Hiβ MidLR lane.
# ≠ Mega MidLR R1015 / ≠ Mega UltraLoLR R1001 / ≠ Ultra HiLR R984 / ≠ SoftCtx MidRank Hiβ Mega MidLR R1027 /
# ≠ MidCtx MidRank Hiβ Ultra MidLR R1017 / ≠ Online / ≠ GRPO
set -euo pipefail
exec >/root/logs/r1032_lean_warm.log 2>&1
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=1,3 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[r1032-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) start GPUs 1,3 ShortCtx MidRank Hiβ Ultra MidLR"
BASE=/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/8e3f1695e058837ed80fec3238ff439fdc2d0f0e
EXP=r1032-vera-offline-dpo-hialpha-midrank-hibeta-shortctx-ultrasuperextrasteps-ep4-midlr
mkdir -p /root/r1032 /root/logs /root/affine_data \
  /root/mining_src/s4-h138-f43-tok-dpo-l2 \
  /root/mining_src/s4-h1-sft \
  /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r1032/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  if [[ -s /root/mining_src/$EXP/dpo_duel_reason.jsonl ]]; then cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r1015/dpo_duel_reason.jsonl ]]; then cp -f /root/r1015/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/mining_src/r1015-vera-offline-dpo-hialpha-midrank-hibeta-shortctx-megasuperextrasteps-ep4-midlr/dpo_duel_reason.jsonl ]]; then
    cp -f /root/mining_src/r1015-vera-offline-dpo-hialpha-midrank-hibeta-shortctx-megasuperextrasteps-ep4-midlr/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r1001/dpo_duel_reason.jsonl ]]; then cp -f /root/r1001/dpo_duel_reason.jsonl "$DATA"
  else echo "FATAL missing Soft Mid Mid Soft data"; exit 1; fi
fi
n=$(wc -l <"$DATA"); echo "[r1032-lean] data_lines=$n"; test "$n" -ge 200
cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
cp -f /root/mining_src/$EXP/merge_lora.py /root/mining_src/s4-h1-sft/merge_lora.py
if [[ -f /root/logs/r1032_train.pid ]]; then
  old=$(cat /root/logs/r1032_train.pid 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "FATAL r1032 already alive pid=$old"; exit 1
  fi
fi
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1,3 | awk '{s+=$1} END{print s+0}')
  echo "[r1032-lean] wait VRAM1+3 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1,3 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo "FATAL GPUs 1,3 busy"; exit 1; }
rm -rf /root/r1032/train; mkdir -p /root/r1032/train
rm -f /root/logs/r1032_train.done /root/logs/r1032_merge.done /root/logs/r1032_merge_ready
: >/root/logs/r1032_train.nohup
nohup python3 /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py \
  --base "$BASE" --data "$DATA" --out-dir /root/r1032/train \
  --max-len 6144 --epochs 4 --lr 1e-6 \
  --lora-r 32 --lora-alpha 128 --beta 0.3 \
  --max-steps 28800 >/root/logs/r1032_train.nohup 2>&1 &
echo $! | tee /root/logs/r1032_train.pid >/root/r1032/train.pid
python3 - <<'PY'
import json,time
from pathlib import Path
meta={
 "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
 "axis":"r1032-vera-offline-dpo-hialpha-midrank-hibeta-shortctx-ultrasuperextrasteps-ep4-midlr",
 "base":"vera6/affine-5g4yy75zuz-t6@8e3f1695",
 "lr":"1e-6","lora_r":32,"lora_alpha":128,"beta":0.3,"max_len":6144,"epochs":4,"max_steps":28800,
 "gpus":"1,3",
 "parent_signal":"R1015 ShortCtx MidRank Hiβ Mega MidLR REFUTE m=-0.002353 ~-0.94× (SE=0.001257 thought✓167 B✓0.403 k=3) → ShortCtx MidRank Hiβ Ultra MidLR isolate; ≠ Mega MidLR R1015 / ≠ Mega UltraLoLR R1001 / ≠ Ultra HiLR R984 / ≠ SoftCtx MidRank Hiβ Mega MidLR R1027 / ≠ MidCtx MidRank Hiβ Ultra MidLR R1017 / ≠ Online / ≠ GRPO",
 "decision_rule":"Stage-5 iff fresh v4 n80 margin>max(2*SE,0.002) AND thought>=80 AND B>=0.30 vs reign36"
}
Path("/root/affine_data/r1032_train_launched.json").write_text(json.dumps(meta,indent=2)+"\n")
print(json.dumps(meta,indent=2))
PY
echo "[r1032-lean] TRAIN launched pid=$(cat /root/logs/r1032_train.pid) BASE=$BASE"
nohup bash /root/mining_src/$EXP/wait_r1032_train_then_merge_p4165.sh >/dev/null 2>&1 &
echo $! >/root/logs/r1032_wait_merge.pid
echo "[r1032-lean] wait→merge armed pid=$(cat /root/logs/r1032_wait_merge.pid)"
nohup bash /root/mining_src/$EXP/wait_r1032_merge_then_n80_p4165.sh >/dev/null 2>&1 &
echo $! >/root/logs/r1032_merge_then_n80.pid
echo "[r1032-lean] merge→n80 waiter pid=$(cat /root/logs/r1032_merge_then_n80.pid)"
