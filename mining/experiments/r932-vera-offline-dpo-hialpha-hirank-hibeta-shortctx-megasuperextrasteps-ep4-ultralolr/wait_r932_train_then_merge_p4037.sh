#!/usr/bin/env bash
set -euo pipefail
exec >>/root/logs/r932_wait_merge.nohup 2>&1
echo "[r932-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) wait train.done"
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
BASE=/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/8e3f1695e058837ed80fec3238ff439fdc2d0f0e
while true; do
  if [[ -f /root/logs/r932_train.done ]]; then break; fi
  pid=$(cat /root/logs/r932_train.pid 2>/dev/null || true)
  if [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] && ! kill -0 "$pid" 2>/dev/null; then
    if [[ -d /root/r932/train ]] && ls /root/r932/train/adapter_model.safetensors >/dev/null 2>&1; then
      echo "[r932-wait] train exited with adapter; marking done"
      date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r932_train.done
      break
    fi
    echo "[r932-wait] FATAL train dead without adapter"; exit 1
  fi
  sleep 30
done
echo "[r932-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) MERGE start"
ADAPTER=/root/r932/train
if [[ -f /root/r932/train/adapter/adapter_config.json ]]; then ADAPTER=/root/r932/train/adapter; fi
mkdir -p /tmp/r932_merged
python3 /root/mining_src/s4-h1-sft/merge_lora.py \
  --base "$BASE" --adapter "$ADAPTER" --out /tmp/r932_merged \
  >/root/logs/r932_merge.nohup 2>&1
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r932_merge.done
echo "[r932-wait] MERGE done $(date -u +%Y-%m-%dT%H:%M:%SZ) adapter=$ADAPTER"
