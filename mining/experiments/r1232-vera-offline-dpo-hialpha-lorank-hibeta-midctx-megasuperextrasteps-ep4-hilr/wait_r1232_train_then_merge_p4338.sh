#!/usr/bin/env bash
set -euo pipefail
LOG=/root/logs/r1232_wait_merge.nohup
exec >>"$LOG" 2>&1
echo "[r1232-merge-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) wait train.done"
while [[ ! -f /root/r1232/train/train.done ]]; do sleep 30; done
echo "[r1232-merge-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) TRAIN_DONE -> merge"
source /root/venv/bin/activate
set -a; source /root/mine.env; set +a
export HF_HOME=/root/hf CUDA_VISIBLE_DEVICES=4,5 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset HF_TOKEN || true
BASE=/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/8e3f1695e058837ed80fec3238ff439fdc2d0f0e
ADAPTER=/root/r1232/train/adapter
MERGE_PY=/root/mining_src/s4-h1-sft/merge_lora.py
rm -rf /tmp/r1232_merged
python3 "$MERGE_PY" --base "$BASE" --adapter "$ADAPTER" --out /tmp/r1232_merged --device-map auto --max-shard-size 5GB
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r1232_merge.done
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r1232_merge_ready
echo "[r1232-merge-wait] MERGE_DONE"
