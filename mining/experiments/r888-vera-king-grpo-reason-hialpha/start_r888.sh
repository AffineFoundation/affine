#!/usr/bin/env bash
# R888: vera6 reign36 king-base × GRPO-on-Reason HiAlpha
# α=128 r=16 G=4 lr=5e-6 @6144 max_steps=1800 epochs=12 kl=0
# ≠ Offline-DPO SoftCtx swarm; ≠ R583 r252 base; base = live king vera6@8e3f1695
# Needs teacher :8000. Prefer TP1 teacher (pod may show 7 GPUs).
set -euo pipefail
export PATH="/root/.local/bin:${PATH}"
# shellcheck disable=SC1091
source /root/venv/bin/activate
if [[ -f /root/mine.env ]]; then set -a; # shellcheck disable=SC1091
  source /root/mine.env; set +a; fi
export HF_HOME=${HF_HOME:-/root/hf}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3}
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
unset HF_TOKEN || true
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1

KING_REV=8e3f1695e058837ed80fec3238ff439fdc2d0f0e
TEACHER_REV=f9a9c5acf5e543cd24d659a056c5dbcda78ffcfc
BASE=${BASE:-/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/$KING_REV}
TEACHER_LOCAL=${TEACHER_LOCAL:-/root/hf/hub/models--zai-org--GLM-4.5-Air-FP8/snapshots/$TEACHER_REV}
SRC=/root/mining_src/r3-reason-grpo
OUT=/root/r888
DATA=$OUT/winner_za_high_l1.jsonl
TRAIN_DIR=$OUT/train
LOG=/root/logs/r888_train.nohup
TEACHER_URL=${TEACHER_URL:-http://127.0.0.1:8000/v1}

mkdir -p "$OUT" /root/logs "$TRAIN_DIR" /root/affine_data
test -d "$BASE" && test -e "$BASE/config.json"
test -s "$DATA"
test -f "$SRC/train_reason_grpo.py"
n=$(wc -l <"$DATA"); test "$n" -ge 200

echo "[r888] wait teacher $TEACHER_URL"
for i in $(seq 1 240); do
  curl -sf --max-time 5 "$TEACHER_URL/models" >/dev/null 2>&1 && break
  (( i == 240 )) && { echo FATAL teacher; exit 1; }
  sleep 15
done

rm -f "$TRAIN_DIR/train.done" "$TRAIN_DIR/train_result.json"
: >"$LOG"
nohup python3 "$SRC/train_reason_grpo.py" \
  --base "$BASE" --data "$DATA" --out-dir "$TRAIN_DIR" \
  --teacher-url "$TEACHER_URL" --teacher-repo "$TEACHER_LOCAL" \
  --max-len 6144 --max-new 512 --epochs 12 --lr 5e-6 \
  --lora-r 16 --lora-alpha 128 --lora-dropout 0.05 \
  --group-size 4 --temperature 0.8 --max-steps 1800 \
  >>"$LOG" 2>&1 &
echo $! | tee /root/logs/r888_train.pid >"$OUT/train.pid"
python3 - <<PY
import json, time
from pathlib import Path
meta = {
  "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
  "hypo": "R888", "pass": 3978, "family": "grpo-teacher-reason",
  "axis": "vera_king_grpo_hialpha_megaextrasteps",
  "pid": int(Path("/root/logs/r888_train.pid").read_text().strip()),
  "base": "$BASE", "base_hub": "vera6/affine-5g4yy75zuz-t6", "base_rev": "$KING_REV",
  "method": "grpo_teacher_reason_hialpha_vera_king",
  "lr": 5e-6, "lora_r": 16, "lora_alpha": 128, "group_size": 4,
  "max_len": 6144, "max_steps": 1800, "epochs": 12,
  "gpus": "$CUDA_VISIBLE_DEVICES",
  "note": "p3978 R888 vera reign36 king×GRPO-Reason HiAlpha; ≠ Offline-DPO; ≠ R583 r252",
}
Path("/root/affine_data/r888_train_launched.json").write_text(json.dumps(meta, indent=2)+"\n")
print(json.dumps(meta, indent=2))
PY
