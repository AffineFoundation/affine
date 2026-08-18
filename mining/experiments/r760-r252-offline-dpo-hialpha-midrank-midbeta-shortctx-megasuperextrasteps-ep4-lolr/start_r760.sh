#!/usr/bin/env bash
# R760: r252 ShortCtx MidRank MidBeta MegaSuperExtra ep4×LoLR
# After R751 Short HiRank MidBeta Mega REFUTE ~−0.25×; amplify R720 SuperExtra ~−0.33×.
# MidRank ⇒ lora_r=32; β=0.1; max_steps=19200; ep=4; @6144.
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
OUT=${OUT:-/root/r760}
DATA=${DATA:-$OUT/dpo_duel_reason.jsonl}
BASE=/root/hf/hub/models--unconst--Affine-5czsc2fc98-r252-merged/snapshots/b42d6245d77fe30885ea8a90387771e1bc465e0f
TRAIN_DIR=$OUT/train
LOG=${LOG:-/root/logs/r760_train.nohup}
LR=${R760_LR:-1e-6}
LORA_R=${R760_LORA_R:-32}
LORA_ALPHA=${R760_LORA_ALPHA:-128}
BETA=${R760_BETA:-0.1}
MAX_STEPS=${R760_MAX_STEPS:-19200}
MAX_LEN=${R760_MAX_LEN:-6144}
EPOCHS=${R760_EPOCHS:-4}
mkdir -p "$OUT" /root/logs "$TRAIN_DIR" /root/affine_data
test -e "$BASE/config.json"
case "$BASE" in *r252-merged*|*5czsc2fc98-r252*) ;; *) echo "FATAL bad BASE"; exit 1 ;; esac
test -s "$DATA"; test -f "$SRC/train_dpo.py"
n=$(wc -l <"$DATA")
echo "[r760] $(date -u +%Y-%m-%dT%H:%M:%SZ) examples=$n lr=$LR r=$LORA_R β=$BETA steps=$MAX_STEPS len=$MAX_LEN ep=$EPOCHS gpus=$CUDA_VISIBLE_DEVICES"
test "$n" -ge 200
rm -f "$TRAIN_DIR/train.done" "$TRAIN_DIR/train_result.json"
nohup python3 "$SRC/train_dpo.py" \
  --base "$BASE" --data "$DATA" --out-dir "$TRAIN_DIR" \
  --max-len "$MAX_LEN" --epochs "$EPOCHS" --lr "$LR" \
  --lora-r "$LORA_R" --lora-alpha "$LORA_ALPHA" --beta "$BETA" \
  --max-steps "$MAX_STEPS" >"$LOG" 2>&1 &
echo $! | tee /root/logs/r760_train.pid >"$OUT/train.pid"
python3 - <<PY
import json, time
from pathlib import Path
meta = {
  "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
  "axis": "r252_offline_dpo_hialpha_midrank_midbeta_shortctx_megasuperextrasteps_ep4_lolr",
  "base": "$BASE", "data": "$DATA", "examples": $n,
  "lr": "$LR", "lora_r": $LORA_R, "lora_alpha": $LORA_ALPHA, "beta": $BETA,
  "max_steps": $MAX_STEPS, "max_len": $MAX_LEN, "epochs": $EPOCHS,
  "gpus": "$CUDA_VISIBLE_DEVICES",
  "pid": int(Path("/root/logs/r760_train.pid").read_text().strip()),
  "parent_signal": "R751 Short HiRank MidBeta Mega REFUTE m=-0.00119 ~-0.25x + R720 Short MidRank MidBeta SuperExtra REFUTE ~-0.33x → MegaSuperExtra ep4 MidRank; ≠ HiRank Mega R751; ≠ SuperExtra R720; ≠ HiRank HiBeta Mega R759; ≠ HiRank LoBeta Mega R756; ≠ Online / ≠ GRPO",
  "decision_rule": "Stage-5 iff fresh n80 paired margin > max(2*SE, 0.002) AND median |z|>=80 AND B pass>=0.30 vs reign35 (v4 k=3)",
}
Path("$OUT/train_meta.json").write_text(json.dumps(meta, indent=2)+"\n")
Path("/root/affine_data/r760_train_launched.json").write_text(json.dumps(meta, indent=2)+"\n")
print(json.dumps(meta, indent=2))
PY
echo "[r760] TRAIN_ARMED pid=$(cat /root/logs/r760_train.pid)"
