#!/usr/bin/env bash
set -euo pipefail
LOG=/root/logs/r923_wait_merge.nohup
exec > >(tee -a "$LOG") 2>&1
echo "[r923-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) wait train.done"
for i in $(seq 1 720); do
  if [[ -f /root/logs/r923_train.done ]] || [[ -f /root/r923/train/train.done ]]; then
    echo "[r923-wait] train.done at iter=$i"; break
  fi
  # also detect adapter
  if ls /root/r923/train/adapter_model.safetensors >/dev/null 2>&1 && ! kill -0 "$(cat /root/logs/r923_train.pid 2>/dev/null)" 2>/dev/null; then
    echo "[r923-wait] adapter+train dead at iter=$i"; touch /root/logs/r923_train.done; break
  fi
  sleep 60
done
[[ -f /root/logs/r923_train.done || -f /root/r923/train/train.done || -f /root/r923/train/adapter_model.safetensors ]] || { echo FATAL no train; exit 1; }
# wait VRAM free on 5,6
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 5,6 | awk '{s+=$1} END{print s+0}')
  [[ "$used" -lt 8192 ]] && break
  sleep 5
done
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
BASE=/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/8e3f1695e058837ed80fec3238ff439fdc2d0f0e
rm -rf /tmp/r923_merged; mkdir -p /tmp/r923_merged
python3 /root/mining_src/s4-h1-sft/merge_lora.py \
  --base "$BASE" --adapter /root/r923/train --out /tmp/r923_merged
touch /root/logs/r923_merge.done
echo "[r923-wait] MERGE done → /tmp/r923_merged"
# handoff note; n80 armed by later pass if TK warm
echo READY_FOR_N80 >/root/logs/r923_merge_ready
