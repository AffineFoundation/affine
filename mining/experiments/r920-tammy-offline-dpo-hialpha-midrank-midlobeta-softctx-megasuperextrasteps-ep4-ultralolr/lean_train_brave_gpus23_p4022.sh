#!/usr/bin/env bash
set -euo pipefail
exec >/root/logs/r920_lean_warm.log 2>&1
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH" PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=2,3 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[r920-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) GPUs 2,3 midlobeta SoftCtx"
BASE=/root/hf/hub/models--tammyfritz--Affine-5hmwhnfbix-tammy2/snapshots/7e5fd5f87e82606c32d59c3d2350e3ddfe49c4b5
EXP=$(basename /home/const/subnet120/mining/experiments/r920-tammy-offline-dpo-hialpha-midrank-midlobeta-softctx-megasuperextrasteps-ep4-ultralolr)
mkdir -p /root/r920 /root/logs /root/affine_data /root/mining_src/s4-h138-f43-tok-dpo-l2 /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r920/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  for src in /root/r906/dpo_duel_reason.jsonl /root/r907/dpo_duel_reason.jsonl /root/r887/dpo_duel_reason.jsonl /root/mining_src/$EXP/dpo_duel_reason.jsonl; do
    [[ -s "$src" ]] && cp -f "$src" "$DATA" && break
  done
fi
n=$(wc -l <"$DATA"); echo "[r920-lean] data_lines=$n"; test "$n" -ge 200
[[ -f /root/mining_src/$EXP/train_dpo.py ]] && cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
if [[ -f /root/logs/r920_train.pid ]]; then
  old=$(cat /root/logs/r920_train.pid 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then echo FATAL alive; exit 1; fi
fi
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2,3 | awk '{s+=$1} END{print s+0}')
  echo "[r920-lean] VRAM=$used iter=$i"; [[ "$used" -lt 8192 ]] && break; sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2,3 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo FATAL busy; exit 1; }
rm -rf /root/r920/train; mkdir -p /root/r920/train
rm -f /root/logs/r920_train.done /root/logs/r920_merge.done /root/logs/r920_merge_launched.p4022 /root/logs/r920_n80_launched.p4022
: >/root/logs/r920_train.nohup
nohup python3 /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py \
  --base "$BASE" --data "$DATA" --out-dir /root/r920/train \
  --max-len 12288 --epochs 4 --lr 5e-7 \
  --lora-r 32 --lora-alpha 128 --beta 0.05 \
  --max-steps 19200 >/root/logs/r920_train.nohup 2>&1 &
echo $! | tee /root/logs/r920_train.pid >/root/r920/train.pid
python3 - <<'PY'
import json,time
from pathlib import Path
meta={
 "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
 "axis":"r920-tammy-offline-dpo-hialpha-midrank-midlobeta-softctx-megasuperextrasteps-ep4-ultralolr",
 "base":"tammyfritz/Affine-5hmwhnfbix-tammy2@7e5fd5f8",
 "lr":"5e-7","lora_r":32,"lora_alpha":128,"beta":0.05,"max_len":12288,"epochs":4,"max_steps":19200,
 "gpus":"2,3",
 "parent_signal":"R906 ShortCtx MidLoβ REFUTE ~−0.27× → SoftCtx MidLoβ transfer; ≠ Online / ≠ GRPO",
 "decision_rule":"Stage-5 iff fresh v4 n80 margin>max(2*SE,0.002) AND thought>=80 AND B>=0.30 vs reign36"
}
Path("/root/affine_data/r920_train_launched.json").write_text(json.dumps(meta,indent=2)+"\n")
print(json.dumps(meta,indent=2))
PY
echo "[r920-lean] TRAIN launched pid=$(cat /root/logs/r920_train.pid)"
