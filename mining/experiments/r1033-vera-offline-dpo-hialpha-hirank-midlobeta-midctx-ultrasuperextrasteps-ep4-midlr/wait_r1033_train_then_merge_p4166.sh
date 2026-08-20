#!/usr/bin/env bash
set -euo pipefail
LOG=/root/logs/r1033_wait_merge.nohup
exec > >(tee -a "$LOG") 2>&1
echo "[r1033-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) wait train.done"
ADAPTER=/root/r1033/train/adapter
for i in $(seq 1 1200); do
  if [[ -f /root/logs/r1033_train.done ]] || [[ -f /root/r1033/train/train.done ]]; then
    echo "[r1033-wait] train.done at iter=$i"; break
  fi
  if [[ -f "$ADAPTER/adapter_model.safetensors" ]] && ! kill -0 "$(cat /root/logs/r1033_train.pid 2>/dev/null)" 2>/dev/null; then
    echo "[r1033-wait] adapter+train dead at iter=$i"; date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r1033_train.done; break
  fi
  sleep 60
done
[[ -f /root/logs/r1033_train.done || -f /root/r1033/train/train.done || -f "$ADAPTER/adapter_model.safetensors" ]] || { echo FATAL no train; exit 1; }
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
  [[ "$used" -lt 8192 ]] && break
  sleep 5
done
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
BASE=/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/8e3f1695e058837ed80fec3238ff439fdc2d0f0e
rm -rf /tmp/r1033_merged
test -f "$ADAPTER/adapter_config.json"
MERGE_PY=/root/mining_src/r1033-vera-offline-dpo-hialpha-hirank-midlobeta-midctx-ultrasuperextrasteps-ep4-midlr/merge_lora.py
[[ -f "$MERGE_PY" ]] || MERGE_PY=/root/mining_src/s4-h1-sft/merge_lora.py
export CUDA_VISIBLE_DEVICES=4,5
python3 "$MERGE_PY" --base "$BASE" --adapter "$ADAPTER" --out /tmp/r1033_merged --device-map auto --max-shard-size 5GB
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r1033_merge.done
echo READY_FOR_N80 >/root/logs/r1033_merge_ready
echo "[r1033-wait] MERGE done → /tmp/r1033_merged"
