#!/usr/bin/env bash
# p4027: R914 ShortCtx Loβ m=+0.000760~0.14× → ShortCtx Hiβ=0.3 isolate on R888 GPUs 5,6
set -euo pipefail
exec >/root/logs/r923_lean_warm.log 2>&1
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=5,6 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[r923-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) start GPUs 5,6 Hiβ ShortCtx"
BASE=/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/8e3f1695e058837ed80fec3238ff439fdc2d0f0e
EXP=r923-vera-offline-dpo-hialpha-midrank-hibeta-shortctx-megasuperextrasteps-ep4-ultralolr
mkdir -p /root/r923 /root/logs /root/affine_data \
  /root/mining_src/s4-h138-f43-tok-dpo-l2 \
  /root/mining_src/s4-h1-sft \
  /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r923/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  if [[ -s /root/r914/dpo_duel_reason.jsonl ]]; then cp -f /root/r914/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/r898/dpo_duel_reason.jsonl ]]; then cp -f /root/r898/dpo_duel_reason.jsonl "$DATA"
  elif [[ -s /root/mining_src/$EXP/dpo_duel_reason.jsonl ]]; then cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl "$DATA"
  else echo "FATAL missing Soft Mid Mid Soft data"; exit 1; fi
fi
n=$(wc -l <"$DATA"); echo "[r923-lean] data_lines=$n"; test "$n" -ge 200
cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
if [[ -f /root/mining_src/$EXP/merge_lora.py ]]; then
  cp -f /root/mining_src/$EXP/merge_lora.py /root/mining_src/s4-h1-sft/merge_lora.py
fi
test -f /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
test -f /root/mining_src/s4-h1-sft/merge_lora.py
if [[ -f /root/logs/r923_train.pid ]]; then
  old=$(cat /root/logs/r923_train.pid 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "FATAL r923 already alive pid=$old"; exit 1
  fi
fi
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 5,6 | awk '{s+=$1} END{print s+0}')
  echo "[r923-lean] wait VRAM5+6 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 5,6 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo "FATAL GPUs 5,6 busy"; exit 1; }
rm -rf /root/r923/train; mkdir -p /root/r923/train
rm -f /root/logs/r923_train.done /root/logs/r923_merge.done
: >/root/logs/r923_train.nohup
nohup python3 /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py \
  --base "$BASE" --data "$DATA" --out-dir /root/r923/train \
  --max-len 6144 --epochs 4 --lr 5e-7 \
  --lora-r 32 --lora-alpha 128 --beta 0.3 \
  --max-steps 19200 >/root/logs/r923_train.nohup 2>&1 &
echo $! | tee /root/logs/r923_train.pid >/root/r923/train.pid
python3 - <<'PY'
import json,time
from pathlib import Path
meta={
 "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
 "axis":"r923-vera-offline-dpo-hialpha-midrank-hibeta-shortctx-megasuperextrasteps-ep4-ultralolr",
 "base":"vera6/affine-5g4yy75zuz-t6@8e3f1695",
 "lr":"5e-7","lora_r":32,"lora_alpha":128,"beta":0.3,"max_len":6144,"epochs":4,"max_steps":19200,
 "gpus":"5,6",
 "parent_signal":"R914 ShortCtx Loβ m=+0.000760 SE=0.002639 bar≈0.00528 ~0.14× thought✓155 B✓0.354 → ShortCtx Hiβ=0.3 isolate; ≠ Loβ R914 / ≠ Midβ R898 / ≠ MidLoβ ShortCtx R913 / ≠ SoftCtx Midβ R861 / ≠ Online / ≠ GRPO",
 "decision_rule":"Stage-5 iff fresh v4 n80 margin>max(2*SE,0.002) AND thought>=80 AND B>=0.30 vs reign36"
}
Path("/root/affine_data/r923_train_launched.json").write_text(json.dumps(meta,indent=2)+"\n")
print(json.dumps(meta,indent=2))
PY
echo "[r923-lean] TRAIN launched pid=$(cat /root/logs/r923_train.pid) BASE=$BASE"
