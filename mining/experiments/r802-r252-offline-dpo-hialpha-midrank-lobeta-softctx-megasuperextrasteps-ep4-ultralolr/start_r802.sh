#!/usr/bin/env bash
# R802: r252 SoftCtx MidRank LoBeta MegaSuperExtra ep4×UltraLoLR (R761 UltraLoLR after R782 REFUTE freed 6,7)
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
OUT=${OUT:-/root/r802}
DATA=${DATA:-$OUT/dpo_duel_reason.jsonl}
BASE=/root/hf/hub/models--unconst--Affine-5czsc2fc98-r252-merged/snapshots/b42d6245d77fe30885ea8a90387771e1bc465e0f
TRAIN_DIR=$OUT/train
LOG=${LOG:-/root/logs/r802_train.nohup}
LR=${R802_LR:-5e-7}
LORA_R=${R802_LORA_R:-32}
LORA_ALPHA=${R802_LORA_ALPHA:-128}
BETA=${R802_BETA:-0.02}
MAX_STEPS=${R802_MAX_STEPS:-19200}
MAX_LEN=${R802_MAX_LEN:-12288}
EPOCHS=${R802_EPOCHS:-4}
mkdir -p "$OUT" /root/logs "$TRAIN_DIR" /root/affine_data
test -e "$BASE/config.json"
case "$BASE" in *r252-merged*|*5czsc2fc98-r252*) ;; *) echo "FATAL bad BASE (want r252)"; exit 1 ;; esac
test -s "$DATA"; test -f "$SRC/train_dpo.py"
n=$(wc -l <"$DATA")
echo "[r802] $(date -u +%Y-%m-%dT%H:%M:%SZ) examples=$n lr=$LR r=$LORA_R β=$BETA steps=$MAX_STEPS len=$MAX_LEN ep=$EPOCHS gpus=$CUDA_VISIBLE_DEVICES base=$BASE"
test "$n" -ge 200
rm -f "$TRAIN_DIR/train.done" "$TRAIN_DIR/train_result.json"
nohup python3 "$SRC/train_dpo.py" \
  --base "$BASE" --data "$DATA" --out-dir "$TRAIN_DIR" \
  --max-len "$MAX_LEN" --epochs "$EPOCHS" --lr "$LR" \
  --lora-r "$LORA_R" --lora-alpha "$LORA_ALPHA" --beta "$BETA" \
  --max-steps "$MAX_STEPS" >"$LOG" 2>&1 &
echo $! | tee /root/logs/r802_train.pid >"$OUT/train.pid"
python3 -c "
import json, time
from pathlib import Path
meta = {
  'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
  'axis': 'r252_offline_dpo_hialpha_midrank_lobeta_softctx_megasuperextrasteps_ep4_ultralolr',
  'base': '$BASE', 'data': '$DATA', 'examples': $n,
  'lr': '$LR', 'lora_r': $LORA_R, 'lora_alpha': $LORA_ALPHA, 'beta': $BETA,
  'max_steps': $MAX_STEPS, 'max_len': $MAX_LEN, 'epochs': $EPOCHS,
  'gpus': '$CUDA_VISIBLE_DEVICES',
  'pid': int(Path('/root/logs/r802_train.pid').read_text().strip()),
  'parent_signal': 'R761 Soft Mid Mid Soft Mega LoLR REFUTE ~0.079x → UltraLoLR; slot freed by R782 Soft Mid Lo Soft Mega LoLR REFUTE ~0.21x (R784 already covers UltraLoLR of R782 on brave); ≠ R786 tammy Soft Mid Mid Soft UltraLoLR; ≠ R791 MidCtx LoBeta UltraLoLR; ≠ Online / ≠ GRPO',
  'decision_rule': 'Stage-5 iff fresh n80 paired margin > max(2*SE, 0.002) AND median |z|>=80 AND B pass>=0.30 vs reign35 (v4 k=3)',
  'n80_path': 'local R252 GPUs 6,7 chall after merge',
}
Path('$OUT/train_meta.json').write_text(json.dumps(meta, indent=2)+'\n')
Path('/root/affine_data/r802_train_launched.json').write_text(json.dumps(meta, indent=2)+'\n')
print(json.dumps(meta, indent=2))
"
echo "[r802] TRAIN_ARMED pid=$(cat /root/logs/r802_train.pid)"
