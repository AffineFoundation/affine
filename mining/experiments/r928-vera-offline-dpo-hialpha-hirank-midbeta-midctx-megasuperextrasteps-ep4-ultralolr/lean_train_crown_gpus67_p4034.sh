#!/usr/bin/env bash
# p4034: R912 MidRank Midβ MidCtx REFUTE ~0.53× → HiRank r=64 Midβ MidCtx isolate
set -euo pipefail
exec >/root/logs/r928_lean_warm.log 2>&1
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH" PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=6,7 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[r928-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) GPUs 6,7 HiRank Midβ MidCtx"
BASE=/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/8e3f1695e058837ed80fec3238ff439fdc2d0f0e
EXP=r928-vera-offline-dpo-hialpha-hirank-midbeta-midctx-megasuperextrasteps-ep4-ultralolr
mkdir -p /root/r928 /root/logs /root/affine_data /root/mining_src/s4-h138-f43-tok-dpo-l2 /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r928/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  if [[ -s /root/r912/dpo_duel_reason.jsonl ]]; then cp -f /root/r912/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/mining_src/dpo_duel_reason.jsonl ]]; then cp -f /root/mining_src/dpo_duel_reason.jsonl "$DATA"
  else echo FATAL missing Soft Mid Mid Soft data; exit 1; fi
fi
n=$(wc -l <"$DATA"); echo "[r928-lean] data_lines=$n"; test "$n" -ge 200
if [[ -f /root/mining_src/$EXP/train_dpo.py ]]; then
  cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
fi
test -f /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
  echo "[r928-lean] VRAM=$used iter=$i"; [[ "$used" -lt 8192 ]] && break; sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo FATAL busy; exit 1; }
rm -rf /root/r928/train; mkdir -p /root/r928/train
rm -f /root/logs/r928_train.done
: >/root/logs/r928_train.nohup
nohup python3 /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py \
  --base "$BASE" --data "$DATA" --out-dir /root/r928/train \
  --max-len 8192 --epochs 4 --lr 5e-7 \
  --lora-r 64 --lora-alpha 128 --beta 0.1 \
  --max-steps 19200 >/root/logs/r928_train.nohup 2>&1 &
echo $! | tee /root/logs/r928_train.pid >/root/r928/train.pid
python3 - <<'PY'
import json,time
from pathlib import Path
meta={
 "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
 "axis":"r928-vera-offline-dpo-hialpha-hirank-midbeta-midctx-megasuperextrasteps-ep4-ultralolr",
 "base":"vera6/affine-5g4yy75zuz-t6@8e3f1695",
 "lr":"5e-7","lora_r":64,"lora_alpha":128,"beta":0.1,"max_len":8192,"epochs":4,"max_steps":19200,
 "gpus":"6,7",
 "parent_signal":"R912 MidRank Midβ MidCtx REFUTE m=+0.005202 SE=0.004877 bar≈0.00975 ~0.53× thought✓167 B✓0.506 → HiRank r=64 Midβ MidCtx isolate; ≠ MidRank R912 / ≠ MidLoβ HiRank MidCtx R925 / ≠ SoftCtx / ≠ Online / ≠ GRPO",
 "decision_rule":"Stage-5 iff fresh v4 n80 margin>max(2*SE,0.002) AND thought>=80 AND B>=0.30 vs reign36"
}
Path("/root/affine_data/r928_train_launched.json").write_text(json.dumps(meta,indent=2)+"\n")
print(json.dumps(meta,indent=2))
PY
echo "[r928-lean] TRAIN launched pid=$(cat /root/logs/r928_train.pid)"
