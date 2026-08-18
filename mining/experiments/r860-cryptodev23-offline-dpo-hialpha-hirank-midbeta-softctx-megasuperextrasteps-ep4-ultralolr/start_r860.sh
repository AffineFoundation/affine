#!/usr/bin/env bash
# R860: cryptoDev23 SoftCtx HiRank MidBeta MegaSuperExtra ep4×UltraLoLR (non-king parent)
# R830 marsplan SoftCtx HiRank Midβ REFUTE m=-0.00113~-0.21× vs reign36 → non-king isolate
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
OUT=${OUT:-/root/r860}
DATA=${DATA:-$OUT/dpo_duel_reason.jsonl}
BASE=/root/hf/hub/models--cryptoDev23--Affine-5Dku3dYp9j-hk8161/snapshots/55b7ffe003d078a8a131673f677b2584548a502e
TRAIN_DIR=$OUT/train
LOG=${LOG:-/root/logs/r860_train.nohup}
LR=${R860_LR:-5e-7}
LORA_R=${R860_LORA_R:-64}
LORA_ALPHA=${R860_LORA_ALPHA:-128}
BETA=${R860_BETA:-0.1}
MAX_STEPS=${R860_MAX_STEPS:-19200}
MAX_LEN=${R860_MAX_LEN:-12288}
EPOCHS=${R860_EPOCHS:-4}
mkdir -p "$OUT" /root/logs "$TRAIN_DIR" /root/affine_data
test -e "$BASE/config.json"
case "$BASE" in *cryptoDev23*|*5Dku3dYp9j*) ;; *) echo "FATAL bad BASE (want cryptoDev23)"; exit 1 ;; esac
test -s "$DATA"; test -f "$SRC/train_dpo.py"
n=$(wc -l <"$DATA")
echo "[r860] $(date -u +%Y-%m-%dT%H:%M:%SZ) examples=$n lr=$LR r=$LORA_R β=$BETA steps=$MAX_STEPS len=$MAX_LEN ep=$EPOCHS gpus=$CUDA_VISIBLE_DEVICES base=$BASE"
test "$n" -ge 200
rm -f "$TRAIN_DIR/train.done" "$TRAIN_DIR/train_result.json"
nohup python3 "$SRC/train_dpo.py" \
  --base "$BASE" --data "$DATA" --out-dir "$TRAIN_DIR" \
  --max-len "$MAX_LEN" --epochs "$EPOCHS" --lr "$LR" \
  --lora-r "$LORA_R" --lora-alpha "$LORA_ALPHA" --beta "$BETA" \
  --max-steps "$MAX_STEPS" >"$LOG" 2>&1 &
echo $! | tee /root/logs/r860_train.pid >"$OUT/train.pid"
python3 -c "
import json, time
from pathlib import Path
meta = {
  'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
  'axis': 'cryptodev23_offline_dpo_hialpha_hirank_midbeta_softctx_megasuperextrasteps_ep4_ultralolr',
  'base': '$BASE', 'data': '$DATA', 'examples': $n,
  'lr': '$LR', 'lora_r': $LORA_R, 'lora_alpha': $LORA_ALPHA, 'beta': $BETA,
  'max_steps': $MAX_STEPS, 'max_len': $MAX_LEN, 'epochs': $EPOCHS,
  'gpus': '$CUDA_VISIBLE_DEVICES',
  'pid': int(Path('/root/logs/r860_train.pid').read_text().strip()),
  'parent_signal': 'R830 marsplan SoftCtx HiRank MidBeta UltraLoLR REFUTE m=-0.001135~-0.21× vs reign36 → cryptoDev23 non-king parent isolate; ≠ R830 marsplan / ≠ Online / ≠ GRPO',
  'decision_rule': 'Stage-5 iff fresh n80 paired margin > max(2*SE, 0.002) AND median |z|>=80 AND B pass>=0.30 vs live king (v4 k=3 tau=0.03)',
  'n80_path': 'local lunar GPUs 6,7 chall after merge vs reign36 vera',
}
Path('$OUT/train_meta.json').write_text(json.dumps(meta, indent=2)+'\n')
Path('/root/affine_data/r860_train_launched.json').write_text(json.dumps(meta, indent=2)+'\n')
print(json.dumps(meta, indent=2))
"
echo "[r860] TRAIN_ARMED pid=$(cat /root/logs/r860_train.pid)"
