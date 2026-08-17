#!/usr/bin/env bash
# R709: marsplan SoftCtx MidRank LoBeta UltraExtra ep3 × LoLR (transfer R637 ~1.45× onto marsplan)
set -euo pipefail
export PATH="/root/.local/bin:${PATH}"
source /root/venv/bin/activate
if [[ -f /root/mine.env ]]; then set -a; source /root/mine.env; set +a; fi
export HF_HOME=${HF_HOME:-/root/hf}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5}
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset HF_TOKEN || true

SRC=${SRC:-/root/mining_src/s4-h138-f43-tok-dpo-l2}
OUT=${OUT:-/root/r709}
DATA=${DATA:-$OUT/dpo_duel_reason.jsonl}
# Pin AFTER mine.env (p3689/p3734) — zesty mine.env BASE is r252
BASE=/root/hf/hub/models--marsplan0624--affine-5gedzafcvg-queen/snapshots/556d02a2adfa9bd42a02de3c766f98be7e44ca46
TRAIN_DIR=$OUT/train
LOG=${LOG:-/root/logs/r709_train.nohup}
LR=${R709_LR:-1e-6}
LORA_R=${R709_LORA_R:-32}
LORA_ALPHA=${R709_LORA_ALPHA:-128}
BETA=${R709_BETA:-0.02}
MAX_STEPS=${R709_MAX_STEPS:-7200}
MAX_LEN=${R709_MAX_LEN:-12288}
EPOCHS=${R709_EPOCHS:-3}

mkdir -p "$OUT" /root/logs "$TRAIN_DIR" /root/affine_data
test -e "$BASE/config.json"
case "$BASE" in *marsplan0624*|*5gedzafcvg*) ;; *) echo "FATAL bad BASE=$BASE"; exit 1 ;; esac
test -s "$DATA"; test -f "$SRC/train_dpo.py"
n=$(wc -l <"$DATA")
echo "[r709] $(date -u +%Y-%m-%dT%H:%M:%SZ) examples=$n lr=$LR r=$LORA_R β=$BETA steps=$MAX_STEPS len=$MAX_LEN ep=$EPOCHS gpus=$CUDA_VISIBLE_DEVICES"
test "$n" -ge 200
rm -f "$TRAIN_DIR/train.done" "$TRAIN_DIR/train_result.json"
nohup python3 "$SRC/train_dpo.py" \
  --base "$BASE" --data "$DATA" --out-dir "$TRAIN_DIR" \
  --max-len "$MAX_LEN" --epochs "$EPOCHS" --lr "$LR" \
  --lora-r "$LORA_R" --lora-alpha "$LORA_ALPHA" --beta "$BETA" \
  --max-steps "$MAX_STEPS" >"$LOG" 2>&1 &
echo $! | tee /root/logs/r709_train.pid >"$OUT/train.pid"
python3 - <<PY
import json, time
from pathlib import Path
meta = {
  "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
  "axis": "marsplan_offline_dpo_hialpha_midrank_lobeta_softctx_ultraextrasteps_ep3_lolr",
  "base": "$BASE", "data": "$DATA", "examples": $n,
  "lr": "$LR", "lora_r": $LORA_R, "lora_alpha": $LORA_ALPHA, "beta": $BETA,
  "max_steps": $MAX_STEPS, "max_len": $MAX_LEN, "epochs": $EPOCHS,
  "gpus": "$CUDA_VISIBLE_DEVICES",
  "pid": int(Path("/root/logs/r709_train.pid").read_text().strip()),
  "parent_signal": "R637 Soft MidRank LoBeta SoftCtx SIGNAL ~1.45× — transfer onto marsplan β=0.02 UltraExtra steps=7200 (≠ R701 marsplan Soft MidRank LoBeta SoftCtx HyperExtra TRAIN; ≠ R703 marsplan Soft MidRank HiBeta SoftCtx HyperExtra REFUTE; ≠ R708 marsplan Soft MidRank MidBeta SoftCtx UltraExtra TRAIN; ≠ R673 r252 Soft MidRank LoBeta SoftCtx UltraExtra REFUTE; ≠ Online / ≠ GRPO)",
  "decision_rule": "Stage-5 iff fresh n80 paired margin > max(2*SE, 0.002) AND median |z|>=80 AND B pass>=0.30 vs reign34 (v4 k=3)",
}
Path("$OUT/train_meta.json").write_text(json.dumps(meta, indent=2)+"\n")
Path("/root/affine_data/r709_train_launched.json").write_text(json.dumps(meta, indent=2)+"\n")
print(json.dumps(meta, indent=2))
PY
echo "[r709] TRAIN_ARMED pid=$(cat /root/logs/r709_train.pid)"
