#!/usr/bin/env bash
# p3869: R804 TRAIN_DONE → LoRA merge on golden GPUs 4,5
set -euo pipefail
source /root/venv/bin/activate
if [[ -f /root/mine.env ]]; then set -a; source /root/mine.env; set +a; fi
export HF_HOME=${HF_HOME:-/root/hf} PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
BASE=/root/hf/hub/models--unconst--Affine-5czsc2fc98-r252-merged/snapshots/b42d6245d77fe30885ea8a90387771e1bc465e0f
ADAPTER=/root/r804/train/adapter
MERGE_DIR=/tmp/r804_merged
GPUS=${GPUS:-4,5}
export CUDA_VISIBLE_DEVICES=$GPUS
LOG=/root/logs/p3869_r804_merge.log
mkdir -p /root/logs /root/affine_data /root/r804/train
: >"$LOG"
log() { echo "[p3869-r804] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }
hub_ok() { local n; n=$(ls "$1"/model-*-of-*.safetensors 2>/dev/null | wc -l || true); [[ -f "$1/config.json" && "${n:-0}" -ge 16 ]]; }
log "START merge GPUs=$GPUS"
test -f "$ADAPTER/adapter_config.json"
test -f "$ADAPTER/adapter_model.safetensors"
date -u +%Y-%m-%dT%H:%M:%SZ > /root/r804/train.done
date -u +%Y-%m-%dT%H:%M:%SZ > /root/r804/train/train.done
rm -rf "$MERGE_DIR"
/root/venv/bin/python3 /root/mining_src/s4-h1-sft/merge_lora.py \
  --base "$BASE" --adapter "$ADAPTER" --out "$MERGE_DIR" \
  --device-map auto --max-shard-size 5GB | tee -a "$LOG"
hub_ok "$MERGE_DIR"
n=$(ls "$MERGE_DIR"/model-*-of-*.safetensors | wc -l)
du -sh "$MERGE_DIR" | tee -a "$LOG"
log "MERGE_DONE shards=$n — next chall n80 vs reign35 (v4 k=3); free 4,5; keep /tmp/r755_merged"
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r804_merge.done
