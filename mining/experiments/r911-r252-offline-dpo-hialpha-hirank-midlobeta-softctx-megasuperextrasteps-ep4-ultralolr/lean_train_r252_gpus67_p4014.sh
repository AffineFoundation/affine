#!/usr/bin/env bash
# p4014: MidCtx Lo/Hiβ REFUTE → SoftCtx MidLoβ=0.05 transfer (≠ MidCtx R899; SoftCtx Midβ R867 flop)
set -euo pipefail
exec >/root/logs/r911_lean_warm.log 2>&1
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH" PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=6,7 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[r911-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) GPUs 6,7 MidLoβ SoftCtx"
BASE=/root/hf/hub/models--unconst--Affine-5czsc2fc98-r252-merged/snapshots/b42d6245d77fe30885ea8a90387771e1bc465e0f
EXP=r911-r252-offline-dpo-hialpha-hirank-midlobeta-softctx-megasuperextrasteps-ep4-ultralolr
mkdir -p /root/r911 /root/logs /root/affine_data /root/mining_src/s4-h138-f43-tok-dpo-l2 /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r911/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  if [[ -s /root/r867/dpo_duel_reason.jsonl ]]; then cp -f /root/r867/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r892/dpo_duel_reason.jsonl ]]; then cp -f /root/r892/dpo_duel_reason.jsonl "$DATA"
  else echo FATAL missing Soft Mid Mid Soft data; exit 1; fi
fi
n=$(wc -l <"$DATA"); echo "[r911-lean] data_lines=$n"; test "$n" -ge 200
if [[ -f /root/mining_src/$EXP/train_dpo.py ]]; then
  cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
fi
test -f /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
if [[ -f /root/logs/r911_train.pid ]]; then
  old=$(cat /root/logs/r911_train.pid 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "FATAL r911 already alive pid=$old"; exit 1
  fi
fi
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
  echo "[r911-lean] VRAM=$used iter=$i"; [[ "$used" -lt 8192 ]] && break; sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo FATAL busy; exit 1; }
rm -rf /root/r911/train; mkdir -p /root/r911/train
rm -f /root/logs/r911_train.done /root/logs/r911_merge.done /root/logs/r911_merge_launched.p4014 /root/logs/r911_n80_launched.p4014
: >/root/logs/r911_train.nohup
nohup python3 /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py \
  --base "$BASE" --data "$DATA" --out-dir /root/r911/train \
  --max-len 12288 --epochs 4 --lr 5e-7 \
  --lora-r 64 --lora-alpha 128 --beta 0.05 \
  --max-steps 19200 >/root/logs/r911_train.nohup 2>&1 &
echo $! | tee /root/logs/r911_train.pid >/root/r911/train.pid
python3 - <<'PY'
import json,time
from pathlib import Path
meta={
 "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
 "axis":"r911-r252-offline-dpo-hialpha-hirank-midlobeta-softctx-megasuperextrasteps-ep4-ultralolr",
 "base":"unconst/Affine-5czsc2fc98-r252-merged@b42d6245",
 "lr":"5e-7","lora_r":64,"lora_alpha":128,"beta":0.05,"max_len":12288,"epochs":4,"max_steps":19200,
 "gpus":"6,7",
 "parent_signal":"R900 MidRank MidLoβ SoftCtx REFUTE m=-0.004297~-0.49x → HiRank MidLoβ SoftCtx isolate; ≠ MidRank R900 / ≠ MidCtx MidLoβ R899 / ≠ ShortCtx MidLoβ R910 / ≠ Online / ≠ GRPO",
 "decision_rule":"Stage-5 iff fresh v4 n80 margin>max(2*SE,0.002) AND thought>=80 AND B>=0.30 vs reign36"
}
Path("/root/affine_data/r911_train_launched.json").write_text(json.dumps(meta,indent=2)+"\n")
print(json.dumps(meta,indent=2))
PY
echo "[r911-lean] TRAIN launched pid=$(cat /root/logs/r911_train.pid)"
