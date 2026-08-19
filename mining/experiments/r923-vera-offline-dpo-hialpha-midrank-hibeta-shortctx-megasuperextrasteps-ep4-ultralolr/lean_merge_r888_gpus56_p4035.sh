#!/usr/bin/env bash
# p4035: R923 merge on R888 GPUs 5,6 after TRAIN_DONE (adapter under train/adapter/)
# Prior wait_r923 used --adapter /root/r923/train → peft miss adapter_config.json.
set -euo pipefail
source /root/venv/bin/activate
if [[ -f /root/mine.env ]]; then set -a; source /root/mine.env; set +a; fi
export HF_HOME=${HF_HOME:-/root/hf} PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
BASE=/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/8e3f1695e058837ed80fec3238ff439fdc2d0f0e
ADAPTER=/root/r923/train/adapter
MERGE_DIR=/tmp/r923_merged
GPUS=${GPUS:-5,6}
export CUDA_VISIBLE_DEVICES=$GPUS
LOG=/root/logs/p4035_r923_merge.log
mkdir -p /root/logs /root/affine_data /root/r923/train
: >"$LOG"
log() { echo "[p4035-r923] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }
hub_ok() { local n; n=$(ls "$1"/model-*-of-*.safetensors 2>/dev/null | wc -l || true); [[ -f "$1/config.json" && "${n:-0}" -ge 16 ]]; }
log "START merge GPUs=$GPUS adapter=$ADAPTER"
test -f "$ADAPTER/adapter_config.json"
test -f "$ADAPTER/adapter_model.safetensors"
date -u +%Y-%m-%dT%H:%M:%SZ > /root/r923/train.done
date -u +%Y-%m-%dT%H:%M:%SZ > /root/r923/train/train.done
rm -rf "$MERGE_DIR"
# Prefer experiment merge_lora if present; else s4-h1-sft
MERGE_PY=/root/mining_src/r923-vera-offline-dpo-hialpha-midrank-hibeta-shortctx-megasuperextrasteps-ep4-ultralolr/merge_lora.py
[[ -f "$MERGE_PY" ]] || MERGE_PY=/root/mining_src/s4-h1-sft/merge_lora.py
/root/venv/bin/python3 "$MERGE_PY" \
  --base "$BASE" --adapter "$ADAPTER" --out "$MERGE_DIR" \
  --device-map auto --max-shard-size 5GB | tee -a "$LOG"
hub_ok "$MERGE_DIR"
n=$(ls "$MERGE_DIR"/model-*-of-*.safetensors | wc -l)
du -sh "$MERGE_DIR" | tee -a "$LOG"
log "MERGE_DONE shards=$n — arm n80 vs reign36 vera (v4 k=3); free 5,6"
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r923_merge.done
echo READY_FOR_N80 >/root/logs/r923_merge_ready
