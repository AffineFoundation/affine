#!/usr/bin/env bash
# p4022: R903 SoftCtx Loβ REFUTE ~−1.66× → MidCtx Loβ=0.02 transfer
set -euo pipefail
exec >/root/logs/r919_lean_warm.log 2>&1
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH" PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=4,5 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[r919-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) GPUs 4,5 Loβ MidCtx"
BASE=/root/hf/hub/models--kevin954--Affine-5DFQbBh8Ev-v5/snapshots/b6575907ea3fe73220d70e0093d473140151406e
EXP=r919-kevin-offline-dpo-hialpha-midrank-lobeta-midctx-megasuperextrasteps-ep4-ultralolr
mkdir -p /root/r919 /root/logs /root/affine_data /root/mining_src/s4-h138-f43-tok-dpo-l2 /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r919/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  if [[ -s /root/r903/dpo_duel_reason.jsonl ]]; then cp -f /root/r903/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r877/dpo_duel_reason.jsonl ]]; then cp -f /root/r877/dpo_duel_reason.jsonl "$DATA"
  else echo FATAL missing Soft Mid Mid Soft data; exit 1; fi
fi
n=$(wc -l <"$DATA"); echo "[r919-lean] data_lines=$n"; test "$n" -ge 200
if [[ -f /root/mining_src/$EXP/train_dpo.py ]]; then
  cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
fi
test -f /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
if [[ -f /root/logs/r919_train.pid ]]; then
  old=$(cat /root/logs/r919_train.pid 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "FATAL r919 already alive pid=$old"; exit 1
  fi
fi
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
  echo "[r919-lean] VRAM=$used iter=$i"; [[ "$used" -lt 8192 ]] && break; sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo FATAL busy; exit 1; }
rm -rf /root/r919/train; mkdir -p /root/r919/train
rm -f /root/logs/r919_train.done /root/logs/r919_merge.done /root/logs/r919_merge_launched.p4022 /root/logs/r919_n80_launched.p4022
: >/root/logs/r919_train.nohup
nohup python3 /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py \
  --base "$BASE" --data "$DATA" --out-dir /root/r919/train \
  --max-len 8192 --epochs 4 --lr 5e-7 \
  --lora-r 32 --lora-alpha 128 --beta 0.02 \
  --max-steps 19200 >/root/logs/r919_train.nohup 2>&1 &
echo $! | tee /root/logs/r919_train.pid >/root/r919/train.pid
python3 - <<'PY'
import json,time
from pathlib import Path
meta={
 "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
 "axis":"r919-kevin-offline-dpo-hialpha-midrank-lobeta-midctx-megasuperextrasteps-ep4-ultralolr",
 "base":"kevin954/Affine-5DFQbBh8Ev-v5@b6575907",
 "lr":"5e-7","lora_r":32,"lora_alpha":128,"beta":0.02,"max_len":8192,"epochs":4,"max_steps":19200,
 "gpus":"4,5",
 "parent_signal":"R903 SoftCtx Loβ REFUTE m=-0.018370~-1.66× thought✓ B✓ → MidCtx Loβ=0.02 transfer; ≠ SoftCtx Loβ R903 / ≠ SoftCtx Hiβ R904 / ≠ SoftCtx Midβ R877 / ≠ Online / ≠ GRPO",
 "decision_rule":"Stage-5 iff fresh v4 n80 margin>max(2*SE,0.002) AND thought>=80 AND B>=0.30 vs reign36"
}
Path("/root/affine_data/r919_train_launched.json").write_text(json.dumps(meta,indent=2)+"\n")
print(json.dumps(meta,indent=2))
PY
echo "[r919-lean] TRAIN launched pid=$(cat /root/logs/r919_train.pid)"
