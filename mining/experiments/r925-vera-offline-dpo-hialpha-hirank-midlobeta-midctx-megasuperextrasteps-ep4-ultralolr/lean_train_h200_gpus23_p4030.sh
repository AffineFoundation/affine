#!/usr/bin/env bash
# p4030: R925 vera MidCtx MidLoβ=0.05 HiRank r=64 Soft Mid Mid Soft UltraLoLR on H200 GPUs 2,3
set -euo pipefail
exec >/root/logs/r925_lean_warm.log 2>&1
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=2,3 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[r925-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) start GPUs 2,3 MidLoβ HiRank MidCtx"
BASE=/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/8e3f1695e058837ed80fec3238ff439fdc2d0f0e
EXP=r925-vera-offline-dpo-hialpha-hirank-midlobeta-midctx-megasuperextrasteps-ep4-ultralolr
mkdir -p /root/r925 /root/logs /root/affine_data \
  /root/mining_src/s4-h138-f43-tok-dpo-l2 \
  /root/mining_src/s4-h1-sft \
  /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r925/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  if [[ -s /root/r924/dpo_duel_reason.jsonl ]]; then cp -f /root/r924/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r886/dpo_duel_reason.jsonl ]]; then cp -f /root/r886/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/mining_src/$EXP/dpo_duel_reason.jsonl ]]; then cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl "$DATA"
  else echo "FATAL missing Soft Mid Mid Soft data"; exit 1; fi
fi
n=$(wc -l <"$DATA"); echo "[r925-lean] data_lines=$n"; test "$n" -ge 200
cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
if [[ -f /root/mining_src/$EXP/merge_lora.py ]]; then
  cp -f /root/mining_src/$EXP/merge_lora.py /root/mining_src/s4-h1-sft/merge_lora.py
fi
test -f /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
test -f /root/mining_src/s4-h1-sft/merge_lora.py
if [[ -f /root/logs/r925_train.pid ]]; then
  old=$(cat /root/logs/r925_train.pid 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "FATAL r925 already alive pid=$old"; exit 1
  fi
fi
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2,3 | awk '{s+=$1} END{print s+0}')
  echo "[r925-lean] wait VRAM2+3 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2,3 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo "FATAL GPUs 2,3 busy"; exit 1; }
rm -rf /root/r925/train; mkdir -p /root/r925/train
rm -f /root/logs/r925_train.done /root/logs/r925_merge.done
: >/root/logs/r925_train.nohup
nohup python3 /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py \
  --base "$BASE" --data "$DATA" --out-dir /root/r925/train \
  --max-len 8192 --epochs 4 --lr 5e-7 \
  --lora-r 64 --lora-alpha 128 --beta 0.05 \
  --max-steps 19200 >/root/logs/r925_train.nohup 2>&1 &
echo $! | tee /root/logs/r925_train.pid >/root/r925/train.pid
python3 - <<'PY'
import json,time
from pathlib import Path
meta={
 "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
 "axis":"r925-vera-offline-dpo-hialpha-hirank-midlobeta-midctx-megasuperextrasteps-ep4-ultralolr",
 "base":"vera6/affine-5g4yy75zuz-t6@8e3f1695",
 "lr":"5e-7","lora_r":64,"lora_alpha":128,"beta":0.05,"max_len":8192,"epochs":4,"max_steps":19200,
 "gpus":"2,3",
 "parent_signal":"R901 MidRank MidCtx MidLoβ m=+0.002527 SE=0.001972 bar≈0.00394 ~0.64× thought✓156 B✓0.438 → HiRank r=64 isolate; ≠ MidRank R901 / ≠ Midβ MidCtx R912 / ≠ Hiβ MidCtx R924 / ≠ SoftCtx HiRank R862 / ≠ Online / ≠ GRPO",
 "decision_rule":"Stage-5 iff fresh v4 n80 margin>max(2*SE,0.002) AND thought>=80 AND B>=0.30 vs reign36"
}
Path("/root/affine_data/r925_train_launched.json").write_text(json.dumps(meta,indent=2)+"\n")
print(json.dumps(meta,indent=2))
PY
echo "[r925-lean] TRAIN launched pid=$(cat /root/logs/r925_train.pid) BASE=$BASE"
