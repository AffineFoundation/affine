#!/usr/bin/env bash
set -euo pipefail
exec >/root/logs/r889_lean_warm.log 2>&1
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf; unset HF_TOKEN || true
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=4,5 SKIP_LOCAL_TKC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[r889-lean] $(date -u +%Y-%m-%dT%H:%M:%SZ) start GPUs 4,5"
BASE=/root/hf/hub/models--tammyfritz--Affine-5hmwhnfbix-tammy2/snapshots/7e5fd5f87e82606c32d59c3d2350e3ddfe49c4b5
EXP=r889-tammy-offline-dpo-hialpha-midrank-midlobeta-midctx-megasuperextrasteps-ep4-ultralolr
mkdir -p /root/r889 /root/logs /root/affine_data /root/mining_src/s4-h138-f43-tok-dpo-l2
test -e "$BASE/config.json"
# Soft Mid Mid Soft data (same md5 as R875/R876/R887)
if [[ ! -s /root/r889/dpo_duel_reason.jsonl ]]; then
  cp -f /root/r875/dpo_duel_reason.jsonl /root/r889/dpo_duel_reason.jsonl
fi
DATA=/root/r889/dpo_duel_reason.jsonl
test -s "$DATA"
n=$(wc -l <"$DATA"); echo "[r889-lean] data_lines=$n"; test "$n" -ge 200
if [[ -f /root/mining_src/$EXP/train_dpo.py ]]; then
  cp -f /root/mining_src/$EXP/train_dpo.py /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
fi
test -f /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py
if [[ -f /root/logs/r889_train.pid ]]; then
  old=$(cat /root/logs/r889_train.pid 2>/dev/null || true)
  if [[ -n "${old:-}" && "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "FATAL r889 already alive pid=$old"; exit 1
  fi
fi
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
  echo "[r889-lean] wait VRAM used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo "FATAL GPUs 4,5 busy"; exit 1; }
rm -rf /root/r889/train; mkdir -p /root/r889/train; : >/root/logs/r889_train.nohup
nohup python3 /root/mining_src/s4-h138-f43-tok-dpo-l2/train_dpo.py \
  --base "$BASE" --data "$DATA" --out-dir /root/r889/train \
  --max-len 8192 --epochs 4 --lr 5e-7 \
  --lora-r 32 --lora-alpha 128 --beta 0.05 \
  --max-steps 19200 >/root/logs/r889_train.nohup 2>&1 &
echo $! | tee /root/logs/r889_train.pid >/root/r889/train.pid
python3 - <<'PY'
import json,time
from pathlib import Path
meta={
 "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
 "axis":"r889-tammy-offline-dpo-hialpha-midrank-midlobeta-midctx-megasuperextrasteps-ep4-ultralolr",
 "base":"tammyfritz/Affine-5hmwhnfbix-tammy2@7e5fd5f8",
 "lr":"5e-7","lora_r":32,"lora_alpha":128,"beta":0.05,"max_len":8192,"epochs":4,"max_steps":19200,
 "gpus":"4,5",
 "parent_signal":"R875 MidCtx MidRank Hiβ ~0.007× REFUTE + R850 MidCtx MidRank Midβ ~0.04× → MidLoβ=0.05 MidCtx Soft Mid Mid Soft UltraLoLR isolate; ≠ SoftCtx R887 / ≠ ShortCtx R890 / ≠ Online / ≠ GRPO",
 "decision_rule":"Stage-5 iff fresh v4 n80 margin>max(2*SE,0.002) AND thought>=80 AND B>=0.30 vs reign36"
}
Path("/root/affine_data/r889_train_launched.json").write_text(json.dumps(meta,indent=2)+"\n")
print(json.dumps(meta,indent=2))
PY
echo "[r889-lean] TRAIN launched pid=$(cat /root/logs/r889_train.pid)"
