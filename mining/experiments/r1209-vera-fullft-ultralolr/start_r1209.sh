#!/usr/bin/env bash
# R1209: Vera FullFT UltraLoLR (R1191 MidLR REFUTE → lr=5e-7 isolate).
set -euo pipefail
export PATH="/root/.local/bin:${PATH}"
source /root/venv/bin/activate
if [[ -f /root/mine.env ]]; then set -a; source /root/mine.env; set +a; fi
export HF_TOKEN="${HF_TOKEN:-}"
export HF_HOME=${HF_HOME:-/root/hf}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

SRC=${SRC:-/root/mining_src/s4-h121-f26-full-ft}
OUT=${OUT:-/root/r1209}
DATA=${DATA:-/root/r1191/winner_za_high_l2.jsonl}
BASE=${BASE:-/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/8e3f1695e058837ed80fec3238ff439fdc2d0f0e}
TRAIN_DIR=$OUT/train
LOG=${LOG:-/root/logs/r1209_train.nohup}
LR=${R1209_LR:-5e-7}
MAX_LEN=${R1209_MAX_LEN:-8192}

mkdir -p "$OUT" /root/logs "$TRAIN_DIR" /root/affine_data /root/h121
ln -sfn "$DATA" /root/h121/winner_za_high_l2.jsonl 2>/dev/null || true
test -d "$BASE"
test -s "$DATA"
test -f "$SRC/train_full.py"
n=$(wc -l <"$DATA")
echo "[r1209] $(date -u +%Y-%m-%dT%H:%M:%SZ) examples=$n lr=$LR max_len=$MAX_LEN gpus=$CUDA_VISIBLE_DEVICES"
test "$n" -ge 200

python3 /root/mining_src/s4-h1v2-sft/verify_thought_mask.py \
  --data "$DATA" --out "$OUT/thought_mask_verify.json"

rm -f "$TRAIN_DIR/train.done" "$TRAIN_DIR/train_result.json"
nohup python3 "$SRC/train_full.py" \
  --base "$BASE" --data "$DATA" --out-dir "$TRAIN_DIR" \
  --max-len "$MAX_LEN" --epochs 1 --lr "$LR" \
  --batch 1 --grad-accum 8 --loss-on thought \
  >"$LOG" 2>&1 &
echo $! | tee /root/logs/r1209_train.pid >"$OUT/train.pid"
# Compat for H121 post_train probes that read h121 pid
cp -f /root/logs/r1209_train.pid /root/logs/h121_train.pid
cp -f /root/logs/r1209_train.pid /root/logs/r1191_train.pid
ln -sfn "$TRAIN_DIR" /root/h121/train 2>/dev/null || true
python3 - <<PY
import json, time
from pathlib import Path
meta = {
    "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "axis": "R1209", "hypo": "R1209", "family": "F26",
    "pid": int(Path("/root/logs/r1209_train.pid").read_text().strip()),
    "base": "$BASE",
    "base_hub": "vera6/affine-5g4yy75zuz-t6",
    "base_rev": "8e3f1695e058837ed80fec3238ff439fdc2d0f0e",
    "data": "$DATA", "examples": $n, "lr": float("$LR"),
    "loss_on": "thought", "recipe": "full_ft_no_lora",
    "max_len": int("$MAX_LEN"), "gpus": "$CUDA_VISIBLE_DEVICES",
    "out": "$TRAIN_DIR", "log": "$LOG",
    "note": "R1209 vera×FullFT UltraLoLR lr=5e-7 @8192 after R1201 HiLR REFUTE ~-0.46x; FullFT Mid+Hi LR exhausted → UltraLoLR isolate; ≠ Offline-DPO / ≠ Online-DPO / ≠ GRPO",
}
Path("/root/affine_data/r1209_train_launched.json").write_text(json.dumps(meta, indent=2)+"\n")
print(json.dumps(meta, indent=2))
print("[r1209] TRAIN_LAUNCHED pid=%s" % meta["pid"])
PY
