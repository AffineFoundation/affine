#!/usr/bin/env bash
# p4000: R898 vera Soft Mid Mid Soft MidRank MidBeta ShortCtx UltraLoLR on R888 GPUs 5,6
# After R861 SoftCtx Midβ LOST ~0.59×δ → ShortCtx transfer on king parent; leave GRPO 2,3 + T0 + K4 alone.
set -euo pipefail
exec >/root/logs/r898_lean_warm.log 2>&1
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=5,6 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[r898-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) start GPUs 5,6"
BASE=/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/8e3f1695e058837ed80fec3238ff439fdc2d0f0e
EXP=r898-vera-offline-dpo-hialpha-midrank-midbeta-shortctx-megasuperextrasteps-ep4-ultralolr
mkdir -p /root/r898 /root/logs /root/affine_data \
  /root/mining_src/s4-h138-f43-tok-dpo-l2 \
  /root/mining_src/s4-h1-sft \
  /root/mining_src/$EXP
test -e "$BASE/config.json"
DATA=/root/r898/dpo_duel_reason.jsonl
if [[ ! -s "$DATA" ]]; then
  if [[ -s /root/mining_src/$EXP/dpo_duel_reason.jsonl ]]; then
    cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl "$DATA"
  else
    echo "FATAL missing Soft Mid Mid Soft data"; exit 1
  fi
fi
n=$(wc -l <"$DATA"); echo "[r898-lean] data_lines=$n"; test "$n" -ge 200
cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
if [[ -f /root/mining_src/$EXP/merge_lora.py ]]; then
  cp -f /root/mining_src/$EXP/merge_lora.py /root/mining_src/s4-h1-sft/merge_lora.py
fi
test -f /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
test -f /root/mining_src/s4-h1-sft/merge_lora.py
if [[ -f /root/logs/r898_train.pid ]]; then
  old=$(cat /root/logs/r898_train.pid 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "FATAL r898 already alive pid=$old"; exit 1
  fi
fi
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 5,6 | awk '{s+=$1} END{print s+0}')
  echo "[r898-lean] wait VRAM5+6 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 5,6 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo "FATAL GPUs 5,6 busy"; exit 1; }
rm -rf /root/r898/train; mkdir -p /root/r898/train; : >/root/logs/r898_train.nohup
nohup python3 /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py \
  --base "$BASE" --data "$DATA" --out-dir /root/r898/train \
  --max-len 6144 --epochs 4 --lr 5e-7 \
  --lora-r 32 --lora-alpha 128 --beta 0.1 \
  --max-steps 19200 >/root/logs/r898_train.nohup 2>&1 &
echo $! | tee /root/logs/r898_train.pid >/root/r898/train.pid
python3 - <<'PY'
import json,time
from pathlib import Path
meta={
 "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
 "axis":"r898-vera-offline-dpo-hialpha-midrank-midbeta-shortctx-megasuperextrasteps-ep4-ultralolr",
 "base":"vera6/affine-5g4yy75zuz-t6@8e3f1695",
 "lr":"5e-7","lora_r":32,"lora_alpha":128,"beta":0.1,"max_len":6144,"epochs":4,"max_steps":19200,
 "gpus":"5,6",
 "parent_signal":"R861 SoftCtx MidRank Midβ LOST m=+0.001182~0.59×δ → ShortCtx@6144 transfer on same king parent Soft Mid Mid Soft UltraLoLR; ≠ SoftCtx R861/R885/R886 / ≠ GRPO R888 / ≠ Online / ≠ MidCtx",
 "decision_rule":"Stage-5 iff fresh v4 n80 margin>max(2*SE,0.002) AND thought>=80 AND B>=0.30 vs reign36"
}
Path("/root/affine_data/r898_train_launched.json").write_text(json.dumps(meta,indent=2)+"\n")
print(json.dumps(meta,indent=2))
PY
echo "[r898-lean] TRAIN launched pid=$(cat /root/logs/r898_train.pid) BASE=$BASE"
