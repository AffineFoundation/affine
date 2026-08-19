#!/usr/bin/env bash
# p3954: r864 TRAIN_DONE → LoRA merge on GPUs 4,5
set -euo pipefail
source /root/venv/bin/activate
if [[ -f /root/mine.env ]]; then set -a; source /root/mine.env; set +a; fi
export HF_HOME=${HF_HOME:-/root/hf} PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
BASE=/root/hf/hub/models--marsplan0624--affine-5gedzafcvg-queen/snapshots/556d02a2adfa9bd42a02de3c766f98be7e44ca46
ADAPTER=/root/r864/train/adapter
MERGE_DIR=/tmp/r864_merged
GPUS=${GPUS:-4,5}
export CUDA_VISIBLE_DEVICES=4,5
LOG=/root/logs/p3954_r864_merge.log
mkdir -p /root/logs /root/affine_data /root/r864/train
: >"$LOG"
log() { echo "[p3954-r864] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }
hub_ok() { local n; n=$(ls "$1"/model-*-of-*.safetensors 2>/dev/null | wc -l || true); [[ -f "$1/config.json" && "${n:-0}" -ge 16 ]]; }
log "START merge GPUs=$GPUS"
test -f "$ADAPTER/adapter_config.json"
test -f "$ADAPTER/adapter_model.safetensors"
date -u +%Y-%m-%dT%H:%M:%SZ > /root/r864/train.done
date -u +%Y-%m-%dT%H:%M:%SZ > /root/r864/train/train.done
rm -rf "$MERGE_DIR"
/root/venv/bin/python3 /root/mining_src/s4-h1-sft/merge_lora.py \
  --base "$BASE" --adapter "$ADAPTER" --out "$MERGE_DIR" \
  --device-map auto --max-shard-size 5GB | tee -a "$LOG"
hub_ok "$MERGE_DIR"
n=$(ls "$MERGE_DIR"/model-*-of-*.safetensors | wc -l)
du -sh "$MERGE_DIR" | tee -a "$LOG"
log "MERGE_DONE shards=$n — stamp for host-relay→lunar n80"
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r864_merge.done
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r864_scp_needed.p3954
