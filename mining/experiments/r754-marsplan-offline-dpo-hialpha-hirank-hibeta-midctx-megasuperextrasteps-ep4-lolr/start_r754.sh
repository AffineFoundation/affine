#!/usr/bin/env bash
# R754: marsplan MidCtx HiRank HiBeta MegaSuperExtra ep4xLoLR (amplify R747 SuperExtra REFUTE ~-0.37x)
set -euo pipefail
export PATH="/root/.local/bin:${PATH}"
source /root/venv/bin/activate
if [[ -f /root/mine.env ]]; then set -a; source /root/mine.env; set +a; fi
# p3799 hard-pin marsplan BASE AFTER mine.env (mine.env BASE=r252 must not win)
BASE=/root/hf/hub/models--marsplan0624--affine-5gedzafcvg-queen/snapshots/556d02a2adfa9bd42a02de3c766f98be7e44ca46
export BASE
export HF_HOME=${HF_HOME:-/root/hf}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6,7}
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset HF_TOKEN || true
SRC=${SRC:-/root/mining_src/s4-h138-f43-tok-dpo-l2}
OUT=${OUT:-/root/r754}
DATA=${DATA:-$OUT/dpo_duel_reason.jsonl}
TRAIN_DIR=$OUT/train
LOG=${LOG:-/root/logs/r754_train.nohup}
LR=${R754_LR:-1e-6}
LORA_R=${R754_LORA_R:-64}
LORA_ALPHA=${R754_LORA_ALPHA:-128}
BETA=${R754_BETA:-0.3}
MAX_STEPS=${R754_MAX_STEPS:-19200}
MAX_LEN=${R754_MAX_LEN:-8192}
EPOCHS=${R754_EPOCHS:-4}
mkdir -p "$OUT" /root/logs "$TRAIN_DIR" /root/affine_data
test -e "$BASE/config.json"
case "$BASE" in *marsplan*|*5gedzafcvg*) ;; *) echo "FATAL bad BASE=$BASE"; exit 1 ;; esac
test -s "$DATA"; test -f "$SRC/train_dpo.py"
n=$(wc -l <"$DATA")
echo "[r754] $(date -u +%Y-%m-%dT%H:%M:%SZ) examples=$n lr=$LR r=$LORA_R beta=$BETA steps=$MAX_STEPS len=$MAX_LEN ep=$EPOCHS gpus=$CUDA_VISIBLE_DEVICES BASE=$BASE"
test "$n" -ge 200
rm -f "$TRAIN_DIR/train.done" "$TRAIN_DIR/train_result.json"
nohup python3 "$SRC/train_dpo.py" \
  --base "$BASE" --data "$DATA" --out-dir "$TRAIN_DIR" \
  --max-len "$MAX_LEN" --epochs "$EPOCHS" --lr "$LR" \
  --lora-r "$LORA_R" --lora-alpha "$LORA_ALPHA" --beta "$BETA" \
  --max-steps "$MAX_STEPS" >"$LOG" 2>&1 &
echo $! | tee /root/logs/r754_train.pid >"$OUT/train.pid"
python3 -c "import json,time; from pathlib import Path; meta={\"utc\":time.strftime(\"%Y-%m-%dT%H:%M:%SZ\",time.gmtime()),\"axis\":\"marsplan_offline_dpo_hialpha_hirank_hibeta_midctx_megasuperextrasteps_ep4_lolr\",\"base\":\"$BASE\",\"data\":\"$DATA\",\"lr\":\"$LR\",\"lora_r\":$LORA_R,\"lora_alpha\":$LORA_ALPHA,\"beta\":$BETA,\"max_steps\":$MAX_STEPS,\"max_len\":$MAX_LEN,\"epochs\":$EPOCHS,\"gpus\":\"$CUDA_VISIBLE_DEVICES\",\"pid\":int(Path(\"/root/logs/r754_train.pid\").read_text().strip()),\"parent_signal\":\"R747 SuperExtra REFUTE m=-0.00346 ~-0.37x -> MegaSuperExtra ep4\"}; Path(\"$OUT/train_meta.json\").write_text(json.dumps(meta,indent=2)+\"\\n\"); Path(\"/root/affine_data/r754_train_launched.json\").write_text(json.dumps(meta,indent=2)+\"\\n\"); print(json.dumps(meta,indent=2))"
echo "[r754] TRAIN_ARMED pid=$(cat /root/logs/r754_train.pid)"
