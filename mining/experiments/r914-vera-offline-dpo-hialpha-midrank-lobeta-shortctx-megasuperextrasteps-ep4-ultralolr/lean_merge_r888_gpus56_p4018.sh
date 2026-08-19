#!/usr/bin/env bash
# p4018: R914 merge on R888 GPUs 5,6 after TRAIN_DONE
set -euo pipefail
source /root/venv/bin/activate
if [[ -f /root/mine.env ]]; then set -a; source /root/mine.env; set +a; fi
export HF_HOME=${HF_HOME:-/root/hf} PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
BASE=/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/8e3f1695e058837ed80fec3238ff439fdc2d0f0e
ADAPTER=/root/r914/train/adapter
MERGE_DIR=/tmp/r914_merged
GPUS=${GPUS:-5,6}
export CUDA_VISIBLE_DEVICES=$GPUS
LOG=/root/logs/p4018_r914_merge.log
mkdir -p /root/logs /root/affine_data /root/r914/train
: >"$LOG"
log() { echo "[p4018-r914] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }
hub_ok() { local n; n=$(ls "$1"/model-*-of-*.safetensors 2>/dev/null | wc -l || true); [[ -f "$1/config.json" && "${n:-0}" -ge 16 ]]; }
log "START merge GPUs=$GPUS"
test -f "$ADAPTER/adapter_config.json"
test -f "$ADAPTER/adapter_model.safetensors"
date -u +%Y-%m-%dT%H:%M:%SZ > /root/r914/train.done
date -u +%Y-%m-%dT%H:%M:%SZ > /root/r914/train/train.done
rm -rf "$MERGE_DIR"
/root/venv/bin/python3 /root/mining_src/s4-h1-sft/merge_lora.py \
  --base "$BASE" --adapter "$ADAPTER" --out "$MERGE_DIR" \
  --device-map auto --max-shard-size 5GB | tee -a "$LOG"
hub_ok "$MERGE_DIR"
n=$(ls "$MERGE_DIR"/model-*-of-*.safetensors | wc -l)
du -sh "$MERGE_DIR" | tee -a "$LOG"
log "MERGE_DONE shards=$n — arm n80 vs reign36 vera (v4 k=3); free 5,6"
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r914_merge.done
