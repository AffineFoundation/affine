#!/usr/bin/env bash
# p4122: R980 SoftCtx HiRank Midβ Ultra HiLR REFUTE ~−1.63× → SoftCtx HiRank Midβ Mega MidLR
# Isolates MidLR (1e-6) between R959 Mega UltraLoLR CROWN_OK and R979 Mega HiLR REFUTE;
# ≠ Ultra HiLR R980 / ≠ Mega HiLR R979 / ≠ Mega UltraLoLR R959 / ≠ Ultra UltraLoLR R947 /
# ≠ SoftCtx HiRank MidLoβ Ultra HiLR R975 / ≠ Online / ≠ GRPO
set -euo pipefail
exec >/root/logs/r992_lean_warm.log 2>&1
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=2,3 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[r992-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) start GPUs 2,3 SoftCtx HiRank Midβ Mega MidLR"
BASE=/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/8e3f1695e058837ed80fec3238ff439fdc2d0f0e
EXP=r992-vera-offline-dpo-hialpha-hirank-midbeta-softctx-megasuperextrasteps-ep4-midlr
mkdir -p /root/r992 /root/logs /root/affine_data \
  /root/mining_src/s4-h138-f43-tok-dpo-l2 \
  /root/mining_src/s4-h1-sft \
  /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r992/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  if [[ -s /root/mining_src/$EXP/dpo_duel_reason.jsonl ]]; then cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r980/dpo_duel_reason.jsonl ]]; then cp -f /root/r980/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r938/dpo_duel_reason.jsonl ]]; then cp -f /root/r938/dpo_duel_reason.jsonl "$DATA"
  else echo "FATAL missing Soft Mid Mid Soft data"; exit 1; fi
fi
n=$(wc -l <"$DATA"); echo "[r992-lean] data_lines=$n"; test "$n" -ge 200
cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
cp -f /root/mining_src/$EXP/merge_lora.py /root/mining_src/s4-h1-sft/merge_lora.py
if [[ -f /root/logs/r992_train.pid ]]; then
  old=$(cat /root/logs/r992_train.pid 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "FATAL r992 already alive pid=$old"; exit 1
  fi
fi
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2,3 | awk '{s+=$1} END{print s+0}')
  echo "[r992-lean] wait VRAM2+3 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2,3 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo "FATAL GPUs 2,3 busy"; exit 1; }
rm -rf /root/r992/train; mkdir -p /root/r992/train
rm -f /root/logs/r992_train.done /root/logs/r992_merge.done
: >/root/logs/r992_train.nohup
nohup python3 /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py \
  --base "$BASE" --data "$DATA" --out-dir /root/r992/train \
  --max-len 12288 --epochs 4 --lr 1e-6 \
  --lora-r 64 --lora-alpha 128 --beta 0.1 \
  --max-steps 19200 >/root/logs/r992_train.nohup 2>&1 &
echo $! | tee /root/logs/r992_train.pid >/root/r992/train.pid
python3 - <<'PY'
import json,time
from pathlib import Path
meta={
 "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
 "axis":"r992-vera-offline-dpo-hialpha-hirank-midbeta-softctx-megasuperextrasteps-ep4-midlr",
 "base":"vera6/affine-5g4yy75zuz-t6@8e3f1695",
 "lr":"1e-6","lora_r":64,"lora_alpha":128,"beta":0.1,"max_len":12288,"epochs":4,"max_steps":19200,
 "gpus":"2,3",
 "parent_signal":"R980 SoftCtx HiRank Midβ Ultra HiLR REFUTE m=-0.007289 ~-1.63× (thought✓188.5 B✓0.4375 k=3) → SoftCtx HiRank Midβ Mega MidLR isolate; ≠ Ultra HiLR R980 / ≠ Mega HiLR R979 / ≠ Mega UltraLoLR R959 CROWN / ≠ Ultra UltraLoLR R947 / ≠ Online / ≠ GRPO",
 "decision_rule":"Stage-5 iff fresh v4 n80 margin>max(2*SE,0.002) AND thought>=80 AND B>=0.30 vs reign36"
}
Path("/root/affine_data/r992_train_launched.json").write_text(json.dumps(meta,indent=2)+"\n")
print(json.dumps(meta,indent=2))
PY
echo "[r992-lean] TRAIN launched pid=$(cat /root/logs/r992_train.pid) BASE=$BASE"
nohup bash /root/mining_src/$EXP/wait_r992_train_then_merge_p4122.sh >/dev/null 2>&1 &
echo $! >/root/logs/r992_wait_merge.pid
echo "[r992-lean] wait→merge armed pid=$(cat /root/logs/r992_wait_merge.pid)"
