#!/usr/bin/env bash
set -euo pipefail
LOG=/root/logs/r1071_wait_merge.nohup
exec > >(tee -a "$LOG") 2>&1
echo "[r1071-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) wait train.done"
ADAPTER=/root/r1071/train/adapter
for i in $(seq 1 1200); do
  if [[ -f /root/logs/r1071_train.done ]] || [[ -f /root/r1071/train/train.done ]]; then echo "[r1071-wait] train.done iter=$i"; break; fi
  if [[ -f "$ADAPTER/adapter_model.safetensors" ]] && ! kill -0 "$(cat /root/logs/r1071_train.pid 2>/dev/null)" 2>/dev/null; then
    echo "[r1071-wait] adapter+train dead"; date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r1071_train.done; break
  fi
  sleep 60
done
[[ -f /root/logs/r1071_train.done || -f /root/r1071/train/train.done || -f "$ADAPTER/adapter_model.safetensors" ]] || { echo FATAL no train; exit 1; }
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
  [[ "$used" -lt 8192 ]] && break; sleep 5
done
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
BASE=/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/8e3f1695e058837ed80fec3238ff439fdc2d0f0e
rm -rf /tmp/r1071_merged
MERGE_PY=/root/mining_src/r1071-vera-offline-dpo-hialpha-hirank-hibeta-shortctx-ultrasuperextrasteps-ep4-hilr/merge_lora.py
[[ -f "$MERGE_PY" ]] || MERGE_PY=/root/mining_src/s4-h1-sft/merge_lora.py
export CUDA_VISIBLE_DEVICES=6,7
python3 "$MERGE_PY" --base "$BASE" --adapter "$ADAPTER" --out /tmp/r1071_merged --device-map auto --max-shard-size 5GB
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r1071_merge.done
echo READY_FOR_N80 >/root/logs/r1071_merge_ready
echo "[r1071-wait] MERGE done"
