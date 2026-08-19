#!/usr/bin/env bash
set -euo pipefail
export PATH="/root/.local/bin:${PATH}"
source /root/venv/bin/activate
[[ -f /root/mine.env ]] && set -a && source /root/mine.env && set +a
export HF_HOME=${HF_HOME:-/root/hf} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5}
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset HF_TOKEN || true
SRC=${SRC:-/root/mining_src/s4-h138-f43-tok-dpo-l2}; OUT=${OUT:-/root/r867}
DATA=${DATA:-$OUT/dpo_duel_reason.jsonl}
BASE=/root/hf/hub/models--unconst--Affine-5czsc2fc98-r252-merged/snapshots/b42d6245d77fe30885ea8a90387771e1bc465e0f
TRAIN_DIR=$OUT/train; LOG=${LOG:-/root/logs/r867_train.nohup}
LR=${R867_LR:-5e-7}; LORA_R=${R867_LORA_R:-32}; LORA_ALPHA=${R867_LORA_ALPHA:-128}
BETA=${R867_BETA:-0.1}; MAX_STEPS=${R867_MAX_STEPS:-19200}; MAX_LEN=${R867_MAX_LEN:-12288}; EPOCHS=${R867_EPOCHS:-4}
mkdir -p "$OUT" /root/logs "$TRAIN_DIR" /root/affine_data
test -e "$BASE/config.json"; case "$BASE" in *r252-merged*) ;; *) echo FATAL; exit 1;; esac
test -s "$DATA"; test -f "$SRC/train_dpo.py"; n=$(wc -l <"$DATA"); test "$n" -ge 200
echo "[r867] $(date -u +%Y-%m-%dT%H:%M:%SZ) n=$n lr=$LR r=$LORA_R beta=$BETA gpus=$CUDA_VISIBLE_DEVICES"
rm -f "$TRAIN_DIR/train.done" "$TRAIN_DIR/train_result.json"
nohup python3 "$SRC/train_dpo.py" --base "$BASE" --data "$DATA" --out-dir "$TRAIN_DIR" --max-len "$MAX_LEN" --epochs "$EPOCHS" --lr "$LR" --lora-r "$LORA_R" --lora-alpha "$LORA_ALPHA" --beta "$BETA" --max-steps "$MAX_STEPS" >"$LOG" 2>&1 &
echo $! | tee /root/logs/r867_train.pid >"$OUT/train.pid"
python3 -c "import json,time;from pathlib import Path;meta=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),axis='r252_offline_dpo_hialpha_midrank_midbeta_softctx_megasuperextrasteps_ep4_ultralolr',base='$BASE',data='$DATA',examples=$n,lr='$LR',lora_r=$LORA_R,lora_alpha=$LORA_ALPHA,beta=$BETA,max_steps=$MAX_STEPS,max_len=$MAX_LEN,epochs=$EPOCHS,gpus='$CUDA_VISIBLE_DEVICES',pid=int(Path('/root/logs/r867_train.pid').read_text().strip()),parent_signal='R844 Soft Mid Mid Soft HiRank MidBeta SoftCtx REFUTE m=-0.00193~-0.19x -> MidRank isolate');Path('$OUT/train_meta.json').write_text(json.dumps(meta,indent=2)+chr(10));Path('/root/affine_data/r867_train_launched.json').write_text(json.dumps(meta,indent=2)+chr(10));print(json.dumps(meta,indent=2))"
echo "[r867] TRAIN_ARMED pid=$(cat /root/logs/r867_train.pid)"
