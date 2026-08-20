#!/usr/bin/env bash
# p4150: R1007 MidCtx MidRank Hiβ Mega MidLR REFUTE m=+0.006593 ~0.76× → MidCtx MidRank Hiβ Ultra MidLR
# Isolates Ultra steps (28800) after Mega MidLR near-miss (mirrors R992→R1012).
# ≠ Mega MidLR R1007 / ≠ Mega UltraLoLR R999 / ≠ Mega HiLR R990 / ≠ SoftCtx Mega UltraLoLR R938 /
# ≠ SoftCtx MidRank Hiβ Mega MidLR R1007 twin / ≠ Online / ≠ GRPO
set -euo pipefail
exec >/root/logs/r1017_lean_warm.log 2>&1
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=6,7 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[r1017-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) start GPUs 6,7 MidCtx MidRank Hiβ Ultra MidLR"
BASE=/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/8e3f1695e058837ed80fec3238ff439fdc2d0f0e
EXP=r1017-vera-offline-dpo-hialpha-midrank-hibeta-midctx-ultrasuperextrasteps-ep4-midlr
mkdir -p /root/r1017 /root/logs /root/affine_data \
  /root/mining_src/s4-h138-f43-tok-dpo-l2 \
  /root/mining_src/s4-h1-sft \
  /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r1017/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  if [[ -s /root/mining_src/$EXP/dpo_duel_reason.jsonl ]]; then cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r1007/dpo_duel_reason.jsonl ]]; then cp -f /root/r1007/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r999/dpo_duel_reason.jsonl ]]; then cp -f /root/r999/dpo_duel_reason.jsonl "$DATA"
  else echo "FATAL missing Soft Mid Mid Soft data"; exit 1; fi
fi
n=$(wc -l <"$DATA"); echo "[r1017-lean] data_lines=$n"; test "$n" -ge 200
cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
cp -f /root/mining_src/$EXP/merge_lora.py /root/mining_src/s4-h1-sft/merge_lora.py
if [[ -f /root/logs/r1017_train.pid ]]; then
  old=$(cat /root/logs/r1017_train.pid 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "FATAL r1017 already alive pid=$old"; exit 1
  fi
fi
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
  echo "[r1017-lean] wait VRAM6+7 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo "FATAL GPUs 6,7 busy"; exit 1; }
rm -rf /root/r1017/train; mkdir -p /root/r1017/train
rm -f /root/logs/r1017_train.done /root/logs/r1017_merge.done /root/logs/r1017_merge_ready
: >/root/logs/r1017_train.nohup
nohup python3 /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py \
  --base "$BASE" --data "$DATA" --out-dir /root/r1017/train \
  --max-len 8192 --epochs 4 --lr 1e-6 \
  --lora-r 32 --lora-alpha 128 --beta 0.3 \
  --max-steps 28800 >/root/logs/r1017_train.nohup 2>&1 &
echo $! | tee /root/logs/r1017_train.pid >/root/r1017/train.pid
python3 - <<'PY'
import json,time
from pathlib import Path
meta={
 "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
 "axis":"r1017-vera-offline-dpo-hialpha-midrank-hibeta-midctx-ultrasuperextrasteps-ep4-midlr",
 "base":"vera6/affine-5g4yy75zuz-t6@8e3f1695",
 "lr":"1e-6","lora_r":32,"lora_alpha":128,"beta":0.3,"max_len":8192,"epochs":4,"max_steps":28800,
 "gpus":"6,7",
 "parent_signal":"R1007 MidCtx MidRank Hiβ Mega MidLR REFUTE m=+0.006593 ~0.76× (thought✓195 B✓0.515 k=3) → MidCtx MidRank Hiβ Ultra MidLR isolate; ≠ Mega MidLR R1007 / ≠ Mega UltraLoLR R999 / ≠ Mega HiLR R990 / ≠ SoftCtx Mega UltraLoLR R938 / ≠ Online / ≠ GRPO",
 "decision_rule":"Stage-5 iff fresh v4 n80 margin>max(2*SE,0.002) AND thought>=80 AND B>=0.30 vs reign36"
}
Path("/root/affine_data/r1017_train_launched.json").write_text(json.dumps(meta,indent=2)+"\n")
print(json.dumps(meta,indent=2))
PY
echo "[r1017-lean] TRAIN launched pid=$(cat /root/logs/r1017_train.pid) BASE=$BASE"
nohup bash /root/mining_src/$EXP/wait_r1017_train_then_merge_p4150.sh >/dev/null 2>&1 &
echo $! >/root/logs/r1017_wait_merge.pid
nohup bash /root/mining_src/$EXP/wait_r1017_merge_then_n80_p4150.sh >/dev/null 2>&1 &
echo $! >/root/logs/r1017_merge_then_n80.pid
echo "[r1017-lean] waiters armed merge=$(cat /root/logs/r1017_wait_merge.pid) n80=$(cat /root/logs/r1017_merge_then_n80.pid)"
