#!/usr/bin/env bash
set -euo pipefail
LOG=/root/logs/r1023_wait_merge.nohup
exec > >(tee -a "$LOG") 2>&1
echo "[r1023-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) wait train.done"
ADAPTER=/root/r1023/train/adapter
for i in $(seq 1 1200); do
  if [[ -f /root/logs/r1023_train.done ]] || [[ -f /root/r1023/train/train.done ]]; then
    echo "[r1023-wait] train.done at iter=$i"; break
  fi
  if [[ -f "$ADAPTER/adapter_model.safetensors" ]] && ! kill -0 "$(cat /root/logs/r1023_train.pid 2>/dev/null)" 2>/dev/null; then
    echo "[r1023-wait] adapter+train dead at iter=$i"; date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r1023_train.done; break
  fi
  sleep 60
done
[[ -f /root/logs/r1023_train.done || -f /root/r1023/train/train.done || -f "$ADAPTER/adapter_model.safetensors" ]] || { echo FATAL no train; exit 1; }
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
  [[ "$used" -lt 8192 ]] && break
  sleep 5
done
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
BASE=/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/8e3f1695e058837ed80fec3238ff439fdc2d0f0e
rm -rf /tmp/r1023_merged
test -f "$ADAPTER/adapter_config.json"
MERGE_PY=/root/mining_src/$(basename /home/const/subnet120/mining/experiments/r1023-vera-offline-dpo-hialpha-midrank-midbeta-softctx-ultrasuperextrasteps-ep4-midlr)/merge_lora.py
[[ -f "$MERGE_PY" ]] || MERGE_PY=/root/mining_src/s4-h1-sft/merge_lora.py
export CUDA_VISIBLE_DEVICES=6,7
python3 "$MERGE_PY" --base "$BASE" --adapter "$ADAPTER" --out /tmp/r1023_merged --device-map auto --max-shard-size 5GB
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r1023_merge.done
echo READY_FOR_N80 >/root/logs/r1023_merge_ready
echo "[r1023-wait] MERGE done → /tmp/r1023_merged"
