#!/usr/bin/env bash
set -euo pipefail
LOG=/root/logs/r976_wait_merge.nohup
exec > >(tee -a "$LOG") 2>&1
echo "[r976-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) wait train.done"
ADAPTER=/root/r976/train/adapter
for i in $(seq 1 1200); do
  if [[ -f /root/logs/r976_train.done ]] || [[ -f /root/r976/train/train.done ]]; then
    echo "[r976-wait] train.done at iter=$i"; break
  fi
  if [[ -f "$ADAPTER/adapter_model.safetensors" ]] && ! kill -0 "$(cat /root/logs/r976_train.pid 2>/dev/null)" 2>/dev/null; then
    echo "[r976-wait] adapter+train dead at iter=$i"; date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r976_train.done; break
  fi
  sleep 60
done
[[ -f /root/logs/r976_train.done || -f /root/r976/train/train.done || -f "$ADAPTER/adapter_model.safetensors" ]] || { echo FATAL no train; exit 1; }
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
  [[ "$used" -lt 8192 ]] && break
  sleep 5
done
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
BASE=/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/8e3f1695e058837ed80fec3238ff439fdc2d0f0e
rm -rf /tmp/r976_merged
test -f "$ADAPTER/adapter_config.json"
MERGE_PY=/root/mining_src/r976-vera-offline-dpo-hialpha-midrank-lobeta-midctx-ultrasuperextrasteps-ep4-hilr/merge_lora.py
[[ -f "$MERGE_PY" ]] || MERGE_PY=/root/mining_src/s4-h1-sft/merge_lora.py
export CUDA_VISIBLE_DEVICES=4,5
python3 "$MERGE_PY" --base "$BASE" --adapter "$ADAPTER" --out /tmp/r976_merged --device-map auto --max-shard-size 5GB
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r976_merge.done
echo READY_FOR_N80 >/root/logs/r976_merge_ready
echo "[r976-wait] MERGE done → /tmp/r976_merged"
