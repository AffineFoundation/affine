#!/usr/bin/env bash
# p4001: R892 MidCtx Loβ REFUTE ~−1.46× + R893 MidCtx Hiβ ~−1.07× → MidCtx MidLoβ=0.05 isolate
set -euo pipefail
exec >/root/logs/r899_lean_warm.log 2>&1
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH" PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=4,5 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[r899-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) GPUs 4,5 MidLoβ MidCtx"
BASE=/root/hf/hub/models--unconst--Affine-5czsc2fc98-r252-merged/snapshots/b42d6245d77fe30885ea8a90387771e1bc465e0f
EXP=r899-r252-offline-dpo-hialpha-midrank-midlobeta-midctx-megasuperextrasteps-ep4-ultralolr
mkdir -p /root/r899 /root/logs /root/affine_data /root/mining_src/s4-h138-f43-tok-dpo-l2 /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r899/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  if [[ -s /root/r892/dpo_duel_reason.jsonl ]]; then cp -f /root/r892/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r872/dpo_duel_reason.jsonl ]]; then cp -f /root/r872/dpo_duel_reason.jsonl "$DATA"
  else echo FATAL missing Soft Mid Mid Soft data; exit 1; fi
fi
n=$(wc -l <"$DATA"); echo "[r899-lean] data_lines=$n"; test "$n" -ge 200
if [[ -f /root/mining_src/$EXP/train_dpo.py ]]; then
  cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
fi
test -f /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
if [[ -f /root/logs/r899_train.pid ]]; then
  old=$(cat /root/logs/r899_train.pid 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "FATAL r899 already alive pid=$old"; exit 1
  fi
fi
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
  echo "[r899-lean] VRAM=$used iter=$i"; [[ "$used" -lt 8192 ]] && break; sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo FATAL busy; exit 1; }
rm -rf /root/r899/train; mkdir -p /root/r899/train
rm -f /root/logs/r899_train.done /root/logs/r899_merge.done /root/logs/r899_merge_launched.p4001 /root/logs/r899_n80_launched.p4001
: >/root/logs/r899_train.nohup
nohup python3 /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py \
  --base "$BASE" --data "$DATA" --out-dir /root/r899/train \
  --max-len 8192 --epochs 4 --lr 5e-7 \
  --lora-r 32 --lora-alpha 128 --beta 0.05 \
  --max-steps 19200 >/root/logs/r899_train.nohup 2>&1 &
echo $! | tee /root/logs/r899_train.pid >/root/r899/train.pid
python3 - <<'PY'
import json,time
from pathlib import Path
meta={
 "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
 "axis":"r899-r252-offline-dpo-hialpha-midrank-midlobeta-midctx-megasuperextrasteps-ep4-ultralolr",
 "base":"unconst/Affine-5czsc2fc98-r252-merged@b42d6245",
 "lr":"5e-7","lora_r":32,"lora_alpha":128,"beta":0.05,"max_len":8192,"epochs":4,"max_steps":19200,
 "gpus":"4,5",
 "parent_signal":"R892 MidCtx Loβ REFUTE m=-0.011624~-1.46x + R893 MidCtx Hiβ m=-0.007590~-1.07x → MidLoβ=0.05 MidCtx isolate; ≠ Loβ R892 / ≠ Hiβ R893 / ≠ Midβ R872 / ≠ SoftCtx / ≠ Online / ≠ GRPO",
 "decision_rule":"Stage-5 iff fresh v4 n80 margin>max(2*SE,0.002) AND thought>=80 AND B>=0.30 vs reign36"
}
Path("/root/affine_data/r899_train_launched.json").write_text(json.dumps(meta,indent=2)+"\n")
print(json.dumps(meta,indent=2))
PY
echo "[r899-lean] TRAIN launched pid=$(cat /root/logs/r899_train.pid)"
