#!/usr/bin/env bash
# p4125: wait R996 train → merge on free GPUs 3,4 (not teacher TP 0,1/5,6) with
# --save-original-format (cryptoDev visual keys; see p4121). Never pkill -f.
set -euo pipefail
LOG=/root/logs/r996_wait_merge.nohup
exec > >(tee -a "$LOG") 2>&1
echo "[r996-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) wait train.done"
ADAPTER=/root/r996/train/adapter
for i in $(seq 1 1200); do
  if [[ -f /root/logs/r996_train.done ]] || [[ -f /root/r996/train/train.done ]]; then
    echo "[r996-wait] train.done at iter=$i"; break
  fi
  if [[ -f "$ADAPTER/adapter_model.safetensors" ]] && ! kill -0 "$(cat /root/logs/r996_train.pid 2>/dev/null)" 2>/dev/null; then
    echo "[r996-wait] adapter+train dead at iter=$i"
    date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r996_train.done
    break
  fi
  sleep 60
done
[[ -f /root/logs/r996_train.done || -f /root/r996/train/train.done || -f "$ADAPTER/adapter_model.safetensors" ]] || { echo FATAL no train; exit 1; }
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3,4 | awk '{s+=$1} END{print s+0}')
  echo "[r996-wait] wait VRAM3+4 used_mib=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 5
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3,4 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo "FATAL GPUs 3,4 busy for merge"; exit 1; }
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
BASE=/root/hf/hub/models--cryptoDev23--Affine-5Dku3dYp9j-hk8161/snapshots/55b7ffe003d078a8a131673f677b2584548a502e
rm -rf /tmp/r996_merged
test -f "$ADAPTER/adapter_config.json"
MERGE_PY=/root/mining_src/r996-cryptodev23-offline-dpo-hialpha-hirank-midbeta-softctx-ultrasuperextrasteps-ep4-ultralolr/merge_lora.py
[[ -f "$MERGE_PY" ]] || MERGE_PY=/root/mining_src/s4-h1-sft/merge_lora.py
export CUDA_VISIBLE_DEVICES=3,4
# cryptoDev: keep HF key layout (model.language_model.* / model.visual.*) — p4121
python3 "$MERGE_PY" --base "$BASE" --adapter "$ADAPTER" --out /tmp/r996_merged \
  --device-map auto --max-shard-size 4GB --save-original-format
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r996_merge.done
echo READY_FOR_N80 >/root/logs/r996_merge_ready
echo "[r996-wait] MERGE done → /tmp/r996_merged"
