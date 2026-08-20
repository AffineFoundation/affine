#!/usr/bin/env bash
# p4180: wait R1051 train → merge on free GPUs 5,6 with --save-original-format (cryptoDev). Never pkill -f.
set -euo pipefail
LOG=/root/logs/r1051_wait_merge.nohup
exec > >(tee -a "$LOG") 2>&1
echo "[r1051-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) wait train.done"
ADAPTER=/root/r1051/train/adapter
for i in $(seq 1 1200); do
  if [[ -f /root/logs/r1051_train.done ]] || [[ -f /root/r1051/train/train.done ]]; then
    echo "[r1051-wait] train.done at iter=$i"; break
  fi
  if [[ -f "$ADAPTER/adapter_model.safetensors" ]] && ! kill -0 "$(cat /root/logs/r1051_train.pid 2>/dev/null)" 2>/dev/null; then
    echo "[r1051-wait] adapter+train dead at iter=$i"
    date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r1051_train.done
    break
  fi
  sleep 60
done
[[ -f /root/logs/r1051_train.done || -f /root/r1051/train/train.done || -f "$ADAPTER/adapter_model.safetensors" ]] || { echo FATAL no train; exit 1; }
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 5,6 | awk '{s+=$1} END{print s+0}')
  echo "[r1051-wait] wait VRAM5+6 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 5
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 5,6 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo "FATAL GPUs 5,6 busy for merge"; exit 1; }
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
BASE=/root/hf/hub/models--cryptoDev23--Affine-5Dku3dYp9j-hk8161/snapshots/55b7ffe003d078a8a131673f677b2584548a502e
rm -rf /tmp/r1051_merged
test -f "$ADAPTER/adapter_config.json"
MERGE_PY=/root/mining_src/r1051-cryptodev23-offline-dpo-hialpha-midrank-midbeta-softctx-megasuperextrasteps-ep4-midlr/merge_lora.py
[[ -f "$MERGE_PY" ]] || MERGE_PY=/root/mining_src/s4-h1-sft/merge_lora.py
export CUDA_VISIBLE_DEVICES=5,6
python3 "$MERGE_PY" --base "$BASE" --adapter "$ADAPTER" --out /tmp/r1051_merged \
  --device-map auto --max-shard-size 4GB --save-original-format
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r1051_merge.done
echo READY_FOR_N80 >/root/logs/r1051_merge_ready
echo "[r1051-wait] MERGE done → /tmp/r1051_merged"
