#!/usr/bin/env bash
# R781: tammy SoftCtx HiRank HiBeta MegaSuperExtra ep4×LoLR (after R693 near-miss + R758 REFUTE)
set -euo pipefail
export PATH="/root/.local/bin:${PATH}"
source /root/venv/bin/activate
if [[ -f /root/mine.env ]]; then set -a; source /root/mine.env; set +a; fi
# hard-pin after mine.env (mine.env BASE=r252 would poison)
export HF_HOME=${HF_HOME:-/root/hf}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3}
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset HF_TOKEN || true
SRC=${SRC:-/root/mining_src/s4-h138-f43-tok-dpo-l2}
OUT=${OUT:-/root/r781}
DATA=${DATA:-$OUT/dpo_duel_reason.jsonl}
BASE=/root/hf/hub/models--tammyfritz--Affine-5hmwhnfbix-tammy2/snapshots/7e5fd5f87e82606c32d59c3d2350e3ddfe49c4b5
TRAIN_DIR=$OUT/train
LOG=${LOG:-/root/logs/r781_train.nohup}
LR=${R781_LR:-1e-6}
LORA_R=${R781_LORA_R:-64}
LORA_ALPHA=${R781_LORA_ALPHA:-128}
BETA=${R781_BETA:-0.3}
MAX_STEPS=${R781_MAX_STEPS:-19200}
MAX_LEN=${R781_MAX_LEN:-12288}
EPOCHS=${R781_EPOCHS:-4}
mkdir -p "$OUT" /root/logs "$TRAIN_DIR" /root/affine_data
test -e "$BASE/config.json"
case "$BASE" in *tammyfritz*|*5hmwhnfbix*|*tammy2*) ;; *) echo "FATAL bad BASE (want tammy king)"; exit 1 ;; esac
test -s "$DATA"; test -f "$SRC/train_dpo.py"
n=$(wc -l <"$DATA")
echo "[r781] $(date -u +%Y-%m-%dT%H:%M:%SZ) examples=$n lr=$LR r=$LORA_R β=$BETA steps=$MAX_STEPS len=$MAX_LEN ep=$EPOCHS gpus=$CUDA_VISIBLE_DEVICES base=$BASE"
test "$n" -ge 200
rm -f "$TRAIN_DIR/train.done" "$TRAIN_DIR/train_result.json"
nohup python3 "$SRC/train_dpo.py" \
  --base "$BASE" --data "$DATA" --out-dir "$TRAIN_DIR" \
  --max-len "$MAX_LEN" --epochs "$EPOCHS" --lr "$LR" \
  --lora-r "$LORA_R" --lora-alpha "$LORA_ALPHA" --beta "$BETA" \
  --max-steps "$MAX_STEPS" >"$LOG" 2>&1 &
echo $! | tee /root/logs/r781_train.pid >"$OUT/train.pid"
python3 -c "
import json, time
from pathlib import Path
meta = {
  'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
  'axis': 'tammy_offline_dpo_hialpha_hirank_hibeta_softctx_megasuperextrasteps_ep4_lolr',
  'base': '$BASE', 'data': '$DATA', 'examples': $n,
  'lr': '$LR', 'lora_r': $LORA_R, 'lora_alpha': $LORA_ALPHA, 'beta': $BETA,
  'max_steps': $MAX_STEPS, 'max_len': $MAX_LEN, 'epochs': $EPOCHS,
  'gpus': '$CUDA_VISIBLE_DEVICES',
  'pid': int(Path('/root/logs/r781_train.pid').read_text().strip()),
  'parent_signal': 'R693 Soft HiRank HiBeta SoftCtx UltraExtra near-miss m=+0.002353 ~0.72x + R758 Mega r252 REFUTE ~-0.46x → king-parent Soft HiRank HiBeta SoftCtx Mega; ≠ Ultra R693; ≠ Mega r252 R758; ≠ Soft Hi Mid Soft tammy R776; ≠ Soft Mid Hi Soft tammy R780; ≠ Online / ≠ GRPO',
  'decision_rule': 'Stage-5 iff fresh n80 paired margin > max(2*SE, 0.002) AND median |z|>=80 AND B pass>=0.30 vs reign35 (v4 k=3)',
  'n80_path': 'host-relay (brave NCCL-blacklist)',
}
Path('$OUT/train_meta.json').write_text(json.dumps(meta, indent=2)+'\n')
Path('/root/affine_data/r781_train_launched.json').write_text(json.dumps(meta, indent=2)+'\n')
print(json.dumps(meta, indent=2))
"
echo "[r781] TRAIN_ARMED pid=$(cat /root/logs/r781_train.pid)"
