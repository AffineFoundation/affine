#!/usr/bin/env bash
# R895: marsplan MidCtx MidRank MidLoBeta Mega UltraLoLR (R835 MidCtx Hi Hi ~−1.19× → MidLoβ MidRank isolate)
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
OUT=${OUT:-/root/r895}
DATA=${DATA:-$OUT/dpo_duel_reason.jsonl}
BASE=/root/hf/hub/models--marsplan0624--affine-5gedzafcvg-queen/snapshots/556d02a2adfa9bd42a02de3c766f98be7e44ca46
TRAIN_DIR=$OUT/train
LOG=${LOG:-/root/logs/r895_train.nohup}
LR=${R895_LR:-5e-7}
LORA_R=${R895_LORA_R:-32}
LORA_ALPHA=${R895_LORA_ALPHA:-128}
BETA=${R895_BETA:-0.3}
MAX_STEPS=${R895_MAX_STEPS:-19200}
MAX_LEN=${R895_MAX_LEN:-8192}
EPOCHS=${R895_EPOCHS:-4}
mkdir -p "$OUT" /root/logs "$TRAIN_DIR" /root/affine_data
test -e "$BASE/config.json"
case "$BASE" in *marsplan0624*|*5gedzafcvg*) ;; *) echo "FATAL bad BASE"; exit 1 ;; esac
if [[ ! -s "$DATA" ]]; then
  cp -f /root/r835/dpo_duel_reason.jsonl "$DATA"
fi
test -s "$DATA"; test -f "$SRC/train_dpo.py"
n=$(wc -l <"$DATA")
echo "[r895] $(date -u +%Y-%m-%dT%H:%M:%SZ) examples=$n lr=$LR r=$LORA_R β=$BETA steps=$MAX_STEPS len=$MAX_LEN ep=$EPOCHS gpus=$CUDA_VISIBLE_DEVICES"
test "$n" -ge 200
rm -rf "$TRAIN_DIR"; mkdir -p "$TRAIN_DIR"
rm -f /root/logs/r895_train.done /root/logs/r895_merge.done /root/logs/r895_merge_launched.p3997
: >"$LOG"
nohup python3 "$SRC/train_dpo.py" \
  --base "$BASE" --data "$DATA" --out-dir "$TRAIN_DIR" \
  --max-len "$MAX_LEN" --epochs "$EPOCHS" --lr "$LR" \
  --lora-r "$LORA_R" --lora-alpha "$LORA_ALPHA" --beta "$BETA" \
  --max-steps "$MAX_STEPS" >"$LOG" 2>&1 &
echo $! | tee /root/logs/r895_train.pid >"$OUT/train.pid"
python3 -c "
import json, time
from pathlib import Path
meta = {
  'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
  'axis': 'marsplan_offline_dpo_hialpha_midrank_hibeta_midctx_megasuperextrasteps_ep4_ultralolr',
  'base': '$BASE', 'data': '$DATA', 'examples': $n,
  'lr': '$LR', 'lora_r': $LORA_R, 'lora_alpha': $LORA_ALPHA, 'beta': $BETA,
  'max_steps': $MAX_STEPS, 'max_len': $MAX_LEN, 'epochs': $EPOCHS,
  'gpus': '$CUDA_VISIBLE_DEVICES',
  'pid': int(Path('/root/logs/r895_train.pid').read_text().strip()),
  'parent_signal': 'R835 MidCtx HiRank Hiβ REFUTE ~-1.19x → MidCtx MidRank Hiβ=0.3 Soft Mid Mid Soft UltraLoLR; ≠ Midβ R796 / ≠ MidLoβ R894 / ≠ HiRank Hiβ R835 / ≠ Loβ R871 / ≠ SoftCtx / ≠ Online / ≠ GRPO',
  'decision_rule': 'Stage-5 iff fresh v4 n80 margin>max(2*SE,0.002) AND thought>=80 AND B>=0.30 vs reign36 (k=3)',
}
Path('$OUT/train_meta.json').write_text(json.dumps(meta, indent=2)+'\n')
Path('/root/affine_data/r895_train_launched.json').write_text(json.dumps(meta, indent=2)+'\n')
print(json.dumps(meta, indent=2))
"
echo "[r895] TRAIN_ARMED pid=$(cat /root/logs/r895_train.pid)"
