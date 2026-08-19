#!/usr/bin/env bash
set -euo pipefail
LOG=/root/logs/r944_wait_merge.nohup
exec > >(tee -a "$LOG") 2>&1
echo "[r944-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) wait train.done"
ADAPTER=/root/r944/train/adapter
for i in $(seq 1 1200); do
  if [[ -f /root/logs/r944_train.done ]] || [[ -f /root/r944/train/train.done ]]; then
    echo "[r944-wait] train.done at iter=$i"; break
  fi
  if [[ -f "$ADAPTER/adapter_model.safetensors" ]] && ! kill -0 "$(cat /root/logs/r944_train.pid 2>/dev/null)" 2>/dev/null; then
    echo "[r944-wait] adapter+train dead at iter=$i"; touch /root/logs/r944_train.done; break
  fi
  sleep 60
done
[[ -f /root/logs/r944_train.done || -f /root/r944/train/train.done || -f "$ADAPTER/adapter_model.safetensors" ]] || { echo FATAL no train; exit 1; }
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0,1 | awk '{s+=$1} END{print s+0}')
  [[ "$used" -lt 8192 ]] && break
  sleep 5
done
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
BASE=/root/hf/hub/models--cryptoDev23--Affine-5Dku3dYp9j-hk8161/snapshots/55b7ffe003d078a8a131673f677b2584548a502e
rm -rf /tmp/r944_merged
test -f "$ADAPTER/adapter_config.json"
MERGE_PY=/root/mining_src/r944-cryptodev23-offline-dpo-hialpha-midrank-midbeta-softctx-ultrasuperextrasteps-ep4-ultralolr/merge_lora.py
[[ -f "$MERGE_PY" ]] || MERGE_PY=/root/mining_src/s4-h1-sft/merge_lora.py
export CUDA_VISIBLE_DEVICES=0,1
python3 "$MERGE_PY" --base "$BASE" --adapter "$ADAPTER" --out /tmp/r944_merged --device-map auto --max-shard-size 5GB
touch /root/logs/r944_merge.done
echo READY_FOR_N80 >/root/logs/r944_merge_ready
echo "[r944-wait] MERGE done → /tmp/r944_merged"
