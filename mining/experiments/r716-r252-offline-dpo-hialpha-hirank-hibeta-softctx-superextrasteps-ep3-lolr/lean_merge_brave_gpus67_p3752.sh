#!/usr/bin/env bash
# p3752: R716 TRAIN_DONE → LoRA merge on brave GPUs 6,7
set -euo pipefail
source /root/venv/bin/activate
if [[ -f /root/mine.env ]]; then set -a; source /root/mine.env; set +a; fi
export HF_HOME=${HF_HOME:-/root/hf} PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
BASE=/root/hf/hub/models--unconst--Affine-5czsc2fc98-r252-merged/snapshots/b42d6245d77fe30885ea8a90387771e1bc465e0f
ADAPTER=/root/r716/train/adapter
MERGE_DIR=/tmp/r716_merged
GPUS=${GPUS:-6,7}
export CUDA_VISIBLE_DEVICES=$GPUS
LOG=/root/logs/p3752_r716_merge.log
mkdir -p /root/logs /root/affine_data /root/r716/train
: >"$LOG"
log() { echo "[p3752-r716] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }
hub_ok() { local n; n=$(ls "$1"/model-*-of-*.safetensors 2>/dev/null | wc -l || true); [[ -f "$1/config.json" && "${n:-0}" -ge 16 ]]; }
log "START merge GPUs=$GPUS"
test -f "$ADAPTER/adapter_config.json"
test -f "$ADAPTER/adapter_model.safetensors"
date -u +%Y-%m-%dT%H:%M:%SZ > /root/r716/train.done
date -u +%Y-%m-%dT%H:%M:%SZ > /root/r716/train/train.done
rm -rf "$MERGE_DIR"
/root/venv/bin/python3 /root/mining_src/s4-h1-sft/merge_lora.py \
  --base "$BASE" --adapter "$ADAPTER" --out "$MERGE_DIR" \
  --device-map auto --max-shard-size 5GB | tee -a "$LOG"
hub_ok "$MERGE_DIR"
n=$(ls "$MERGE_DIR"/model-*-of-*.safetensors | wc -l)
du -sh "$MERGE_DIR" | tee -a "$LOG"
log "MERGE_DONE shards=$n — next chall n80 vs reign34 (v4 k=3); free 6,7; keep /tmp/r696_merged /tmp/r699_merged /tmp/r700_merged /tmp/r704_merged"
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r716_merge.done
