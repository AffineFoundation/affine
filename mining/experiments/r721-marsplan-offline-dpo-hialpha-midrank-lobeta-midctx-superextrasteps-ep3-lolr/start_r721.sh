#!/usr/bin/env bash
# R721: marsplan Soft MidRank LoBeta MidCtx SuperExtra ep3×LoLR
set -euo pipefail
export PATH="/root/.local/bin:${PATH}"
source /root/venv/bin/activate
if [[ -f /root/mine.env ]]; then set -a; source /root/mine.env; set +a; fi
export HF_HOME=${HF_HOME:-/root/hf}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6,7}
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset HF_TOKEN || true
SRC=${SRC:-/root/mining_src/s4-h138-f43-tok-dpo-l2}
OUT=${OUT:-/root/r721}
DATA=${DATA:-$OUT/dpo_duel_reason.jsonl}
BASE=${BASE:-/root/hf/hub/models--marsplan0624--affine-5gedzafcvg-queen/snapshots/556d02a2adfa9bd42a02de3c766f98be7e44ca46}
TRAIN_DIR=$OUT/train
LOG=${LOG:-/root/logs/r721_train.nohup}
LR=${R721_LR:-1e-6}
LORA_R=${R721_LORA_R:-32}
LORA_ALPHA=${R721_LORA_ALPHA:-128}
BETA=${R721_BETA:-0.02}
MAX_STEPS=${R721_MAX_STEPS:-14400}
MAX_LEN=${R721_MAX_LEN:-8192}
EPOCHS=${R721_EPOCHS:-3}
mkdir -p "$OUT" /root/logs "$TRAIN_DIR" /root/affine_data
test -e "$BASE/config.json"
case "$BASE" in *marsplan*|*5gedzafcvg*) ;; *) echo "FATAL bad BASE=$BASE"; exit 1 ;; esac
test -s "$DATA"; test -f "$SRC/train_dpo.py"
n=$(wc -l <"$DATA")
echo "[r721] $(date -u +%Y-%m-%dT%H:%M:%SZ) examples=$n lr=$LR r=$LORA_R β=$BETA steps=$MAX_STEPS len=$MAX_LEN ep=$EPOCHS gpus=$CUDA_VISIBLE_DEVICES"
test "$n" -ge 200
rm -f "$TRAIN_DIR/train.done" "$TRAIN_DIR/train_result.json"
nohup python3 "$SRC/train_dpo.py" \
  --base "$BASE" --data "$DATA" --out-dir "$TRAIN_DIR" \
  --max-len "$MAX_LEN" --epochs "$EPOCHS" --lr "$LR" \
  --lora-r "$LORA_R" --lora-alpha "$LORA_ALPHA" --beta "$BETA" \
  --max-steps "$MAX_STEPS" >"$LOG" 2>&1 &
echo $! | tee /root/logs/r721_train.pid >"$OUT/train.pid"
python3 - <<PY
import json, time
from pathlib import Path
meta = {
  "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
  "axis": "marsplan_offline_dpo_hialpha_midrank_lobeta_midctx_superextrasteps_ep3_lolr",
  "base": "$BASE", "data": "$DATA", "examples": $n,
  "lr": "$LR", "lora_r": $LORA_R, "lora_alpha": $LORA_ALPHA, "beta": $BETA,
  "max_steps": $MAX_STEPS, "max_len": $MAX_LEN, "epochs": $EPOCHS,
  "gpus": "$CUDA_VISIBLE_DEVICES",
  "pid": int(Path("/root/logs/r721_train.pid").read_text().strip()),
  "parent_signal": "R637 Soft MidRank LoBeta SIGNAL ~1.45x + R692 MidCtx MidRank LoBeta UltraExtra → marsplan MidCtx SuperExtra steps=14400 (≠ SoftCtx R712; ≠ UltraExtra 7200 R692; ≠ MidBeta MidCtx R719; ≠ Online / ≠ GRPO)",
  "decision_rule": "Stage-5 iff fresh n80 paired margin > max(2*SE, 0.002) AND median |z|>=80 AND B pass>=0.30 vs reign34 (v4 k=3)",
}
Path("$OUT/train_meta.json").write_text(json.dumps(meta, indent=2)+"\n")
Path("/root/affine_data/r721_train_launched.json").write_text(json.dumps(meta, indent=2)+"\n")
print(json.dumps(meta, indent=2))
PY
echo "[r721] TRAIN_ARMED pid=$(cat /root/logs/r721_train.pid)"
