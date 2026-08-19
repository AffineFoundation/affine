#!/usr/bin/env bash
# p4047: R337 online-DPO HiLR REFUTE m=-0.009386 ~-0.95× → Offline SoftCtx MidRank Midβ UltraExtraSteps(28800) UltraLoLR isolate
# ≠ Online R337 / ≠ SoftCtx Midβ MidRank R861 (19200) / ≠ SoftCtx Hiβ R938 / ≠ SoftCtx MidLoβ R886 / ≠ GRPO
set -euo pipefail
exec >/root/logs/r939_lean_warm.log 2>&1
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=4,5 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[r939-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) start GPUs 4,5 SoftCtx Midβ UltraExtra"
BASE=/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/8e3f1695e058837ed80fec3238ff439fdc2d0f0e
EXP=r939-vera-offline-dpo-hialpha-midrank-midbeta-softctx-ultrasuperextrasteps-ep4-ultralolr
mkdir -p /root/r939 /root/logs /root/affine_data \
  /root/mining_src/s4-h138-f43-tok-dpo-l2 \
  /root/mining_src/s4-h1-sft \
  /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r939/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  if [[ -s /root/mining_src/$EXP/dpo_duel_reason.jsonl ]]; then cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r886/dpo_duel_reason.jsonl ]]; then cp -f /root/r886/dpo_duel_reason.jsonl "$DATA"
  else echo "FATAL missing Soft Mid Mid Soft data"; exit 1; fi
fi
n=$(wc -l <"$DATA"); echo "[r939-lean] data_lines=$n"; test "$n" -ge 200
cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
cp -f /root/mining_src/$EXP/merge_lora.py /root/mining_src/s4-h1-sft/merge_lora.py
test -f /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
if [[ -f /root/logs/r939_train.pid ]]; then
  old=$(cat /root/logs/r939_train.pid 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "FATAL r939 already alive pid=$old"; exit 1
  fi
fi
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
  echo "[r939-lean] wait VRAM4+5 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo "FATAL GPUs 4,5 busy"; exit 1; }
rm -rf /root/r939/train; mkdir -p /root/r939/train
rm -f /root/logs/r939_train.done /root/logs/r939_merge.done
: >/root/logs/r939_train.nohup
nohup python3 /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py \
  --base "$BASE" --data "$DATA" --out-dir /root/r939/train \
  --max-len 12288 --epochs 4 --lr 5e-7 \
  --lora-r 32 --lora-alpha 128 --beta 0.1 \
  --max-steps 28800 >/root/logs/r939_train.nohup 2>&1 &
echo $! | tee /root/logs/r939_train.pid >/root/r939/train.pid
python3 - <<'PY'
import json,time
from pathlib import Path
meta={
 "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
 "axis":"r939-vera-offline-dpo-hialpha-midrank-midbeta-softctx-ultrasuperextrasteps-ep4-ultralolr",
 "base":"vera6/affine-5g4yy75zuz-t6@8e3f1695",
 "lr":"5e-7","lora_r":32,"lora_alpha":128,"beta":0.1,"max_len":12288,"epochs":4,"max_steps":28800,
 "gpus":"4,5",
 "parent_signal":"R337 online-DPO HiLR REFUTE m=-0.009386 SE=0.004937 bar≈0.00987 ~-0.95× thought✓183 B✓0.414 → Offline SoftCtx MidRank Midβ UltraExtra(28800) isolate; ≠ Online R337 / ≠ SoftCtx Midβ R861(19200) / ≠ SoftCtx Hiβ R938 / ≠ SoftCtx MidLoβ R886 / ≠ SoftCtx HiRank Midβ R937 / ≠ GRPO",
 "decision_rule":"Stage-5 iff fresh v4 n80 margin>max(2*SE,0.002) AND thought>=80 AND B>=0.30 vs reign36"
}
Path("/root/affine_data/r939_train_launched.json").write_text(json.dumps(meta,indent=2)+"\n")
print(json.dumps(meta,indent=2))
PY
echo "[r939-lean] TRAIN launched pid=$(cat /root/logs/r939_train.pid) BASE=$BASE"
