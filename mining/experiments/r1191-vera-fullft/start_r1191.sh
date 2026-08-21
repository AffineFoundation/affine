#!/usr/bin/env bash
# R1191: Vera-init thought-only full-FT (R4/H121 method; no LoRA).
# Overlay: upload_and_launch copies this to s4-h121-f26-full-ft/start_h121.sh.
# R1191: Vera-FullFT (≠ Offline-DPO LoRA fleet / ≠ Online-DPO marsplan / ≠ R1158 Reason-GRPO / ≠ R226 genesis FullFT)
set -euo pipefail

export PATH="/root/.local/bin:${PATH}"
# shellcheck disable=SC1091
source /root/venv/bin/activate
if [[ -f /root/mine.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source /root/mine.env
  set +a
fi
export HF_TOKEN="${HF_TOKEN:-}"
export HF_HOME=${HF_HOME:-/root/hf}
# Full FT shards across all GPUs.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

SRC=${SRC:-/root/mining_src/s4-h121-f26-full-ft}
OUT=${OUT:-/root/r1191}
DATA=${DATA:-$OUT/winner_za_high_l2.jsonl}
BASE=${BASE:-/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/8e3f1695e058837ed80fec3238ff439fdc2d0f0e}
TRAIN_DIR=$OUT/train
LOG=${LOG:-/root/logs/r1191_train.nohup}
LR=${R1191_LR:-1e-6}
MAX_LEN=${R1191_MAX_LEN:-8192}

mkdir -p "$OUT" /root/logs "$TRAIN_DIR" /root/affine_data /root/h121 /root/r1191
# H121 train paths still expect /root/h121 data symlink for some tooling.
ln -sfn "$DATA" /root/h121/winner_za_high_l2.jsonl 2>/dev/null || true
test -d "$BASE"
test -s "$DATA"
test -f "$SRC/train_full.py"
test -f /root/mining_src/s4-h1v2-sft/thought_mask.py
n=$(wc -l <"$DATA")
echo "[r1191] $(date -u +%Y-%m-%dT%H:%M:%SZ) examples=$n data=$DATA base=$BASE gpus=$CUDA_VISIBLE_DEVICES"
echo "[r1191] knobs lr=$LR max_len=$MAX_LEN loss_on=thought axis=${R1191_AXIS:-vera_fullft}"
test "$n" -ge 200

echo "[r1191] verify thought cuts (CPU)"
python3 /root/mining_src/s4-h1v2-sft/verify_thought_mask.py \
  --data "$DATA" \
  --out "$OUT/thought_mask_verify.json"

rm -f "$TRAIN_DIR/train.done" "$TRAIN_DIR/train_result.json"
echo "[r1191] $(date -u +%Y-%m-%dT%H:%M:%SZ) launch full-FT lr=$LR on GPUs $CUDA_VISIBLE_DEVICES"
nohup python3 "$SRC/train_full.py" \
  --base "$BASE" \
  --data "$DATA" \
  --out-dir "$TRAIN_DIR" \
  --max-len "$MAX_LEN" \
  --epochs 1 \
  --lr "$LR" \
  --batch 1 \
  --grad-accum 8 \
  --loss-on thought \
  >"$LOG" 2>&1 &
echo $! | tee /root/logs/r1191_train.pid >"$OUT/train.pid"
# Compat pid for H121-era post_train probes.
cp -f /root/logs/r1191_train.pid /root/logs/h121_train.pid
ln -sfn "$TRAIN_DIR" /root/h121/train 2>/dev/null || true
python3 - <<PY
import json, time
from pathlib import Path
meta = {
    "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "axis": "R1191",
    "hypo": "R1191",
    "family": "F26",
    "pid": int(Path("/root/logs/r1191_train.pid").read_text().strip()),
    "base": "$BASE",
    "base_hub": "vera6/affine-5g4yy75zuz-t6",
    "base_rev": "8e3f1695e058837ed80fec3238ff439fdc2d0f0e",
    "data": "$DATA",
    "examples": $n,
    "lr": float("$LR"),
    "loss_on": "thought",
    "recipe": "full_ft_no_lora",
    "max_len": int("$MAX_LEN"),
    "gpus": "$CUDA_VISIBLE_DEVICES",
    "out": "$TRAIN_DIR",
    "log": "$LOG",
    "note": "R1191 vera-init×FullFT thought-only lr=$LR @8192 — ≠ Offline-DPO LoRA fleet / ≠ R1158 GRPO / ≠ Online-DPO",
}
Path("/root/affine_data/r1191_train_launched.json").write_text(json.dumps(meta, indent=2) + "\n")
Path("/root/affine_data/h121_train_launched.json").write_text(json.dumps(meta, indent=2) + "\n")
print(json.dumps(meta, indent=2))
print("[r1191] TRAIN_LAUNCHED pid=%s" % meta["pid"])
PY
