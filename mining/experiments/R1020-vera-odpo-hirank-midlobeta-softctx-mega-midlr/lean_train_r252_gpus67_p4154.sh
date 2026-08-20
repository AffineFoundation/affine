#!/usr/bin/env bash
# p4154: R998 SoftCtx HiRank MidLoβ Mega UltraLoLR REFUTE m=+0.001298 ~0.33× → SoftCtx HiRank MidLoβ Mega MidLR
# Isolates MidLR (1e-6) between Mega UltraLoLR R998 (~0.33×) and prior Ultra HiLR R975 (~-0.52×).
# ≠ Mega UltraLoLR R998 / ≠ Ultra HiLR R975 / ≠ Ultra UltraLoLR R951 / ≠ SoftCtx HiRank Midβ Mega UltraLoLR R959 LOST /
# ≠ SoftCtx MidRank MidLoβ Mega MidLR R1009 / ≠ SoftCtx HiRank Midβ Mega MidLR R992 / ≠ Online / ≠ GRPO
set -euo pipefail
exec >/root/logs/r1020_lean_warm.log 2>&1
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=6,7 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[r1020-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) start GPUs 6,7 SoftCtx HiRank MidLoβ Mega MidLR"
BASE=/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/8e3f1695e058837ed80fec3238ff439fdc2d0f0e
EXP=r1020-vera-offline-dpo-hialpha-hirank-midlobeta-softctx-megasuperextrasteps-ep4-midlr
mkdir -p /root/r1020 /root/logs /root/affine_data \
  /root/mining_src/s4-h138-f43-tok-dpo-l2 \
  /root/mining_src/s4-h1-sft \
  /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r1020/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  if [[ -s /root/mining_src/$EXP/dpo_duel_reason.jsonl ]]; then cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r998/dpo_duel_reason.jsonl ]]; then cp -f /root/r998/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r975/dpo_duel_reason.jsonl ]]; then cp -f /root/r975/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r942/dpo_duel_reason.jsonl ]]; then cp -f /root/r942/dpo_duel_reason.jsonl "$DATA"
  else echo "FATAL missing Soft Mid Mid Soft data"; exit 1; fi
fi
n=$(wc -l <"$DATA"); echo "[r1020-lean] data_lines=$n"; test "$n" -ge 200
cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
cp -f /root/mining_src/$EXP/merge_lora.py /root/mining_src/s4-h1-sft/merge_lora.py
if [[ -f /root/logs/r1020_train.pid ]]; then
  old=$(cat /root/logs/r1020_train.pid 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "FATAL r1020 already alive pid=$old"; exit 1
  fi
fi
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
  echo "[r1020-lean] wait VRAM6+7 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo "FATAL GPUs 6,7 busy"; exit 1; }
rm -rf /root/r1020/train; mkdir -p /root/r1020/train
rm -f /root/logs/r1020_train.done /root/logs/r1020_merge.done /root/logs/r1020_merge_ready
: >/root/logs/r1020_train.nohup
nohup python3 /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py \
  --base "$BASE" --data "$DATA" --out-dir /root/r1020/train \
  --max-len 12288 --epochs 4 --lr 1e-6 \
  --lora-r 64 --lora-alpha 128 --beta 0.05 \
  --max-steps 19200 >/root/logs/r1020_train.nohup 2>&1 &
echo $! | tee /root/logs/r1020_train.pid >/root/r1020/train.pid
python3 - <<'PY'
import json,time
from pathlib import Path
meta={
 "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
 "axis":"r1020-vera-offline-dpo-hialpha-hirank-midlobeta-softctx-megasuperextrasteps-ep4-midlr",
 "base":"vera6/affine-5g4yy75zuz-t6@8e3f1695",
 "lr":"1e-6","lora_r":64,"lora_alpha":128,"beta":0.05,"max_len":12288,"epochs":4,"max_steps":19200,
 "gpus":"6,7",
 "parent_signal":"R998 SoftCtx HiRank MidLoβ Mega UltraLoLR REFUTE m=+0.001298 ~0.33× (thought✓171.5 B✓0.4125 k=3) → SoftCtx HiRank MidLoβ Mega MidLR isolate; ≠ Mega UltraLoLR R998 / ≠ Ultra HiLR R975 / ≠ Ultra UltraLoLR R951 / ≠ SoftCtx HiRank Midβ Mega UltraLoLR R959 LOST / ≠ SoftCtx MidRank MidLoβ Mega MidLR R1009 / ≠ SoftCtx HiRank Midβ Mega MidLR R992 / ≠ Online / ≠ GRPO",
 "decision_rule":"Stage-5 iff fresh v4 n80 margin>max(2*SE,0.002) AND thought>=80 AND B>=0.30 vs reign36"
}
Path("/root/affine_data/r1020_train_launched.json").write_text(json.dumps(meta,indent=2)+"\n")
print(json.dumps(meta,indent=2))
PY
echo "[r1020-lean] TRAIN launched pid=$(cat /root/logs/r1020_train.pid) BASE=$BASE"
nohup bash /root/mining_src/$EXP/wait_r1020_train_then_merge_p4154.sh >/dev/null 2>&1 &
echo $! >/root/logs/r1020_wait_merge.pid
echo "[r1020-lean] wait→merge armed pid=$(cat /root/logs/r1020_wait_merge.pid)"
nohup bash /root/mining_src/$EXP/wait_r1020_merge_then_n80_p4154.sh >/dev/null 2>&1 &
echo $! >/root/logs/r1020_merge_then_n80.pid
echo "[r1020-lean] merge→n80 waiter pid=$(cat /root/logs/r1020_merge_then_n80.pid)"
