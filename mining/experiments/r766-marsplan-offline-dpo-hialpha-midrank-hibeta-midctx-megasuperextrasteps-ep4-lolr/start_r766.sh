#!/usr/bin/env bash
# R766: marsplan MidCtx MidRank HiBeta MegaSuperExtra ep4×LoLR
# (R754 HiRank HiBeta MidCtx Mega REFUTE m=-0.00862 ~−0.91× → MidRank sibling; amplify R723/R733)
set -euo pipefail
export PATH="/root/.local/bin:${PATH}"
source /root/venv/bin/activate
if [[ -f /root/mine.env ]]; then set -a; source /root/mine.env; set +a; fi
BASE=/root/hf/hub/models--marsplan0624--affine-5gedzafcvg-queen/snapshots/556d02a2adfa9bd42a02de3c766f98be7e44ca46
export BASE
export HF_HOME=${HF_HOME:-/root/hf}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6,7}
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset HF_TOKEN || true
SRC=${SRC:-/root/mining_src/s4-h138-f43-tok-dpo-l2}
OUT=${OUT:-/root/r766}
DATA=${DATA:-$OUT/dpo_duel_reason.jsonl}
TRAIN_DIR=$OUT/train
LOG=${LOG:-/root/logs/r766_train.nohup}
LR=${R766_LR:-1e-6}
LORA_R=${R766_LORA_R:-32}
LORA_ALPHA=${R766_LORA_ALPHA:-128}
BETA=${R766_BETA:-0.3}
MAX_STEPS=${R766_MAX_STEPS:-19200}
MAX_LEN=${R766_MAX_LEN:-8192}
EPOCHS=${R766_EPOCHS:-4}
mkdir -p "$OUT" /root/logs "$TRAIN_DIR" /root/affine_data
test -e "$BASE/config.json"
case "$BASE" in *marsplan*|*5gedzafcvg*) ;; *) echo "FATAL bad BASE=$BASE"; exit 1 ;; esac
test -s "$DATA"; test -f "$SRC/train_dpo.py"
n=$(wc -l <"$DATA")
echo "[r766] $(date -u +%Y-%m-%dT%H:%M:%SZ) examples=$n lr=$LR r=$LORA_R beta=$BETA steps=$MAX_STEPS len=$MAX_LEN ep=$EPOCHS gpus=$CUDA_VISIBLE_DEVICES BASE=$BASE"
test "$n" -ge 200
rm -f "$TRAIN_DIR/train.done" "$TRAIN_DIR/train_result.json"
nohup python3 "$SRC/train_dpo.py" \
  --base "$BASE" --data "$DATA" --out-dir "$TRAIN_DIR" \
  --max-len "$MAX_LEN" --epochs "$EPOCHS" --lr "$LR" \
  --lora-r "$LORA_R" --lora-alpha "$LORA_ALPHA" --beta "$BETA" \
  --max-steps "$MAX_STEPS" >"$LOG" 2>&1 &
echo $! | tee /root/logs/r766_train.pid >"$OUT/train.pid"
python3 - <<PY
import json, time
from pathlib import Path
meta = {
  "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
  "axis": "marsplan_offline_dpo_hialpha_midrank_hibeta_midctx_megasuperextrasteps_ep4_lolr",
  "base": "$BASE", "data": "$DATA", "examples": $n,
  "lr": "$LR", "lora_r": $LORA_R, "lora_alpha": $LORA_ALPHA, "beta": $BETA,
  "max_steps": $MAX_STEPS, "max_len": $MAX_LEN, "epochs": $EPOCHS,
  "gpus": "$CUDA_VISIBLE_DEVICES",
  "pid": int(Path("/root/logs/r766_train.pid").read_text().strip()),
  "parent_signal": "R754 MidCtx HiRank HiBeta Mega REFUTE m=-0.00862 ~-0.91x → MidRank sibling MegaSuperExtra ep4; amplify R723/R733 MidRank HiBeta MidCtx; ≠ HiRank Mega R754 / ≠ Soft MidBeta Mega R757 / ≠ MidRank LoBeta Mega R765 / ≠ Online / ≠ GRPO",
  "decision_rule": "Stage-5 iff fresh n80 paired margin > max(2*SE, 0.002) AND median |z|>=80 AND B pass>=0.30 vs reign35 (v4 k=3)",
}
Path("$OUT/train_meta.json").write_text(json.dumps(meta, indent=2)+"\n")
Path("/root/affine_data/r766_train_launched.json").write_text(json.dumps(meta, indent=2)+"\n")
print(json.dumps(meta, indent=2))
PY
echo "[r766] TRAIN_ARMED pid=$(cat /root/logs/r766_train.pid)"
