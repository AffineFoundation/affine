#!/usr/bin/env bash
set -euo pipefail
LOG=/root/logs/r1230_wait_merge.nohup
exec >>"$LOG" 2>&1
echo "[r1230-merge-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) wait train.done"
while [[ ! -f /root/r1230/train/train.done ]]; do sleep 30; done
echo "[r1230-merge-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) TRAIN_DONE -> merge"
source /root/venv/bin/activate
set -a; source /root/mine.env; set +a
export HF_HOME=/root/hf CUDA_VISIBLE_DEVICES=3,4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset HF_TOKEN || true
BASE=/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/8e3f1695e058837ed80fec3238ff439fdc2d0f0e
ADAPTER=/root/r1230/train/adapter
MERGE_PY=/root/mining_src/s4-h1-sft/merge_lora.py
rm -rf /tmp/r1230_merged
python3 "$MERGE_PY" --base "$BASE" --adapter "$ADAPTER" --out /tmp/r1230_merged --device-map auto --max-shard-size 5GB
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r1230_merge.done
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r1230_merge_ready
echo "[r1230-merge-wait] MERGE_DONE"
