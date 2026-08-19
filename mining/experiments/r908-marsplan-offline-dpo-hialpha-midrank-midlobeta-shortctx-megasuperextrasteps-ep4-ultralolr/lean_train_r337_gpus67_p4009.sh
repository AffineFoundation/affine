#!/usr/bin/env bash
set -euo pipefail
exec >/root/logs/r908_lean_warm.log 2>&1
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=6,7 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[r908-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) start GPUs 6,7"
BASE=/root/hf/hub/models--marsplan0624--affine-5gedzafcvg-queen/snapshots/556d02a2adfa9bd42a02de3c766f98be7e44ca46
EXP=r908-marsplan-offline-dpo-hialpha-midrank-midlobeta-shortctx-megasuperextrasteps-ep4-ultralolr
mkdir -p /root/r908 /root/logs /root/affine_data /root/mining_src/s4-h138-f43-tok-dpo-l2 /root/mining_src/$EXP
test -e "$BASE/config.json"
if [[ ! -s /root/r908/dpo_duel_reason.jsonl ]]; then
  if [[ -s /root/r894/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r894/dpo_duel_reason.jsonl /root/r908/dpo_duel_reason.jsonl
  elif [[ -s /root/mining_src/$EXP/dpo_duel_reason.jsonl ]]; then
    cp -f /root/mining_src/$EXP/dpo_duel_reason.jsonl /root/r908/dpo_duel_reason.jsonl
  else
    cp -f /root/r835/dpo_duel_reason.jsonl /root/r908/dpo_duel_reason.jsonl
  fi
fi
DATA=/root/r908/dpo_duel_reason.jsonl
test -s "$DATA"
n=$(wc -l <"$DATA"); echo "[r908-lean] data_lines=$n"; test "$n" -ge 200
test -f /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
if [[ -f /root/logs/r908_train.pid ]]; then
  old=$(cat /root/logs/r908_train.pid 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "FATAL r908 already alive pid=$old"; exit 1
  fi
fi
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
  echo "[r908-lean] wait VRAM used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo "FATAL GPUs 6,7 busy"; exit 1; }
rm -rf /root/r908/train; mkdir -p /root/r908/train
rm -f /root/logs/r908_train.done /root/logs/r908_merge.done /root/logs/r908_merge_launched.p4009
: >/root/logs/r908_train.nohup
nohup python3 /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py \
  --base "$BASE" --data "$DATA" --out-dir /root/r908/train \
  --max-len 6144 --epochs 4 --lr 5e-7 \
  --lora-r 32 --lora-alpha 128 --beta 0.05 \
  --max-steps 19200 >/root/logs/r908_train.nohup 2>&1 &
echo $! | tee /root/logs/r908_train.pid >/root/r908/train.pid
python3 - <<'PY'
import json,time
from pathlib import Path
meta={
 "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
 "axis":"r908-marsplan-offline-dpo-hialpha-midrank-midlobeta-shortctx-megasuperextrasteps-ep4-ultralolr",
 "base":"marsplan0624/affine-5gedzafcvg-queen@556d02a2",
 "lr":"5e-7","lora_r":32,"lora_alpha":128,"beta":0.05,"max_len":6144,"epochs":4,"max_steps":19200,
 "gpus":"6,7",
 "parent_signal":"R894 MidCtx MidRank MidLoβ ~−0.78× REFUTE → ShortCtx MidLoβ=0.05 Soft Mid Mid Soft UltraLoLR isolate; ≠ MidCtx R894 / ≠ Hiβ MidCtx R895 / ≠ SoftCtx R882 / ≠ Online / ≠ GRPO",
 "decision_rule":"Stage-5 iff fresh v4 n80 margin>max(2*SE,0.002) AND thought>=80 AND B>=0.30 vs reign36"
}
Path("/root/affine_data/r908_train_launched.json").write_text(json.dumps(meta,indent=2)+"\n")
print(json.dumps(meta,indent=2))
PY
echo "[r908-lean] TRAIN launched pid=$(cat /root/logs/r908_train.pid)"
