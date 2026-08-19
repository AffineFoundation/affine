#!/usr/bin/env bash
# R338: Marsplan-init online-DPO BigG×HiLR on live teacher Reason (G=8, lr=2e-5, BT vs frozen base).
# Overlay: upload_and_launch copies this to s4-h139-f44-tok-online-dpo-l2/start_h139.sh.
# R338: Marsplan-init online DPO BigG×HiLR G=8 lr=2e-5 (≠ R334/R336/R229 @5e-6 / ≠ R335 BoN-BigG / ≠ R230 BoN / ≠ R231 KL / ≠ R232 Teacher-ZC / ≠ R225–R227; n80 king=marsplan-queen).
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
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6,7}
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}

SRC=${SRC:-/root/mining_src/s4-h139-f44-tok-online-dpo-l2}
OUT=${OUT:-/root/r338}
DATA=${DATA:-$OUT/winner_za_high_l1.jsonl}
BASE=${BASE:-/root/hf/hub/models--marsplan0624--affine-5gedzafcvg-queen/snapshots/8b365bdcd8f270c61fe633fcc95d536a93516e02}
TRAIN_DIR=$OUT/train
LOG=${LOG:-/root/logs/r338_train.nohup}
TEACHER_URL=${TEACHER_URL:-http://127.0.0.1:8000/v1}

LR=${R338_LR:-2e-5}
LORA_R=${R338_LORA_R:-16}
LORA_ALPHA=${R338_LORA_ALPHA:-32}
BETA=${R338_BETA:-0.1}
GROUP=${R338_GROUP:-8}
MAX_STEPS=${R338_MAX_STEPS:-300}
MIN_GAP=${R338_MIN_GAP:-0.0}
TEMP=${R338_TEMP:-1.2}

mkdir -p "$OUT" /root/logs "$TRAIN_DIR" /root/affine_data /root/r338
test -d "$BASE"
test -s "$DATA"
test -f "$SRC/train_online_dpo.py"
n=$(wc -l <"$DATA")
echo "[r338] $(date -u +%Y-%m-%dT%H:%M:%SZ) examples=$n data=$DATA base=$BASE gpus=$CUDA_VISIBLE_DEVICES"
echo "[r338] knobs lr=$LR r=$LORA_R/α$LORA_ALPHA β=$BETA G=$GROUP temp=$TEMP steps=$MAX_STEPS min_gap=$MIN_GAP axis=${R338_AXIS:-marsplan_online_dpo_bigg_hilr}"
test "$n" -ge 200

echo "[r338] wait teacher at $TEACHER_URL"
for i in $(seq 1 240); do
  if curl -sf --max-time 5 "$TEACHER_URL/models" >/dev/null 2>&1; then
    echo "[r338] teacher ready after ${i}×15s"
    break
  fi
  if (( i == 240 )); then
    echo "[r338] FATAL: teacher never came up"
    exit 1
  fi
  sleep 15
done

python - <<'PY'
import importlib
for m in ("peft", "accelerate", "torch", "transformers", "httpx"):
    importlib.import_module(m)
print("deps OK")
PY

rm -f "$TRAIN_DIR/train.done" "$TRAIN_DIR/train_result.json"
echo "[r338] $(date -u +%Y-%m-%dT%H:%M:%SZ) launch online-DPO-teacher-Reason lr=$LR r=$LORA_R G=$GROUP temp=$TEMP on GPUs $CUDA_VISIBLE_DEVICES"
nohup python3 "$SRC/train_online_dpo.py" \
  --base "$BASE" \
  --data "$DATA" \
  --out-dir "$TRAIN_DIR" \
  --teacher-url http://127.0.0.1:8000/v1 \
  --teacher-repo zai-org/GLM-4.5-Air-FP8 \
  --max-len 6144 \
  --max-new 256 \
  --epochs 1 \
  --lr "$LR" \
  --lora-r "$LORA_R" \
  --lora-alpha "$LORA_ALPHA" \
  --beta "$BETA" \
  --group-size "$GROUP" \
  --temperature "$TEMP" \
  --max-steps "$MAX_STEPS" \
  --min-gap "$MIN_GAP" \
  >"$LOG" 2>&1 &
echo $! | tee /root/logs/r338_train.pid >"$OUT/train.pid"
python3 - <<PY
import json, time
from pathlib import Path
meta = {
    "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "hypo": "R338",
    "family": "online-dpo-reason",
    "axis": "${R338_AXIS:-marsplan_online_dpo_bigg_hilr}",
    "pid": int(Path("/root/logs/r338_train.pid").read_text().strip()),
    "base": "$BASE",
    "base_hub": "marsplan0624/affine-5gedzafcvg-queen",
    "base_rev": "8b365bdcd8f270c61fe633fcc95d536a93516e02",
    "data": "$DATA",
    "examples": $n,
    "lr": float("$LR"),
    "method": "online_dpo_teacher_reason",
    "lora_r": int("$LORA_R"),
    "lora_alpha": int("$LORA_ALPHA"),
    "beta": float("$BETA"),
    "group_size": int("$GROUP"),
    "temperature": float("$TEMP"),
    "max_steps": int("$MAX_STEPS"),
    "min_gap": float("$MIN_GAP"),
    "gpus": "$CUDA_VISIBLE_DEVICES",
    "teacher_url": "$TEACHER_URL",
    "out": "$TRAIN_DIR",
    "log": "$LOG",
    "note": "R338 marsplan online DPO: sample G=$GROUP temp=$TEMP min_gap=$MIN_GAP, teacher-Reason labels, BT vs frozen base — ≠ R204–R228 / ≠ R11 Tok / ≠ R228 Offline; p2947 diversify after gap=0 collapse",
}
Path("/root/affine_data/r338_train_launched.json").write_text(json.dumps(meta, indent=2) + "\n")
Path("/root/affine_data/r338_train_launched.json").write_text(json.dumps(meta, indent=2) + "\n")
print(json.dumps(meta, indent=2))
print("[r338] TRAIN_LAUNCHED pid=%s" % meta["pid"])
PY
