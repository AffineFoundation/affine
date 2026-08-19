#!/usr/bin/env bash
# p3991: R872 MidCtx Midβ REFUTE ~−0.93× → MidCtx Hiβ isolate on R252 GPUs 6,7
set -euo pipefail
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH" PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=6,7 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[r893-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) GPUs 6,7"
BASE=/root/hf/hub/models--unconst--Affine-5czsc2fc98-r252-merged/snapshots/b42d6245d77fe30885ea8a90387771e1bc465e0f
EXP=r893-r252-offline-dpo-hialpha-midrank-hibeta-midctx-megasuperextrasteps-ep4-ultralolr
mkdir -p /root/r893 /root/logs /root/affine_data /root/mining_src/s4-h138-f43-tok-dpo-l2 /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r893/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  if [[ -s /root/r872/dpo_duel_reason.jsonl ]]; then cp -f /root/r872/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r844/dpo_duel_reason.jsonl ]]; then cp -f /root/r844/dpo_duel_reason.jsonl "$DATA"
  else echo FATAL missing Soft Mid Mid Soft data; exit 1; fi
fi
n=$(wc -l <"$DATA"); test "$n" -ge 200
cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
  echo "[r893-lean] VRAM=$used iter=$i"; [[ "$used" -lt 8192 ]] && break; sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo FATAL busy; exit 1; }
rm -rf /root/r893/train; mkdir -p /root/r893/train; : >/root/logs/r893_train.nohup
nohup python3 /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py \
  --base "$BASE" --data "$DATA" --out-dir /root/r893/train \
  --max-len 8192 --epochs 4 --lr 5e-7 \
  --lora-r 32 --lora-alpha 128 --beta 0.3 \
  --max-steps 19200 >/root/logs/r893_train.nohup 2>&1 &
echo $! | tee /root/logs/r893_train.pid >/root/r893/train.pid
python3 - <<'PY'
import json,time
from pathlib import Path
meta={
 "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
 "axis":"r893-r252-offline-dpo-hialpha-midrank-hibeta-midctx-megasuperextrasteps-ep4-ultralolr",
 "base":"unconst/Affine-5czsc2fc98-r252-merged@b42d6245",
 "lr":"5e-7","lora_r":32,"lora_alpha":128,"beta":0.3,"max_len":8192,"epochs":4,"max_steps":19200,
 "gpus":"6,7",
 "parent_signal":"R872 MidCtx MidRank Midβ REFUTE m=-0.007333~-0.93x → Hiβ isolate; ≠ Midβ R872 / ≠ SoftCtx R841 / ≠ ShortCtx R873 / ≠ Online / ≠ GRPO",
 "decision_rule":"Stage-5 iff fresh v4 n80 margin>max(2*SE,0.002) AND thought>=80 AND B>=0.30 vs reign36"
}
Path("/root/affine_data/r893_train_launched.json").write_text(json.dumps(meta,indent=2)+"\n")
print(json.dumps(meta,indent=2))
PY
echo "[r893-lean] TRAIN launched pid=$(cat /root/logs/r893_train.pid)"
