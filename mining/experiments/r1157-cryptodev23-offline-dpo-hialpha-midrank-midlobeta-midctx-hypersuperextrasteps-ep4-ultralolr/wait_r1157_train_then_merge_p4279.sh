#!/usr/bin/env bash
set -euo pipefail
LOG=/root/logs/r1157_wait_merge.nohup
exec > >(tee -a "$LOG") 2>&1
echo "[r1157-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) wait train.done"
ADAPTER=/root/r1157/train/adapter
for i in $(seq 1 1200); do
  if [[ -f /root/logs/r1157_train.done || -f /root/r1157/train/train.done ]]; then break; fi
  if [[ -f "$ADAPTER/adapter_model.safetensors" ]] && ! kill -0 "$(cat /root/logs/r1157_train.pid 2>/dev/null)" 2>/dev/null; then
    date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r1157_train.done; break
  fi
  sleep 60
done
[[ -f /root/logs/r1157_train.done || -f /root/r1157/train/train.done || -f "$ADAPTER/adapter_model.safetensors" ]] || { echo FATAL; exit 1; }
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3,4 | awk '{s+=$1} END{print s+0}')
  [[ "$used" -lt 8192 ]] && break; sleep 5
done
set -a; source /root/mine.env; set +a; source /root/venv/bin/activate
export HF_HOME=/root/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
BASE=/root/hf/hub/models--cryptoDev23--Affine-5Dku3dYp9j-hk8161/snapshots/55b7ffe003d078a8a131673f677b2584548a502e
rm -rf /tmp/r1157_merged
MERGE_PY=/root/mining_src/r1157-cryptodev23-offline-dpo-hialpha-midrank-midlobeta-midctx-hypersuperextrasteps-ep4-ultralolr/merge_lora.py
[[ -f "$MERGE_PY" ]] || MERGE_PY=/root/mining_src/s4-h1-sft/merge_lora.py
export CUDA_VISIBLE_DEVICES=3,4
python3 "$MERGE_PY" --base "$BASE" --adapter "$ADAPTER" --out /tmp/r1157_merged --device-map auto --max-shard-size 5GB
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r1157_merge.done
echo READY >/root/logs/r1157_merge_ready
