#!/usr/bin/env bash
set -euo pipefail
exec >>/root/logs/r931_wait_merge.nohup 2>&1
echo "[r931-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) wait train.done"
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
BASE=/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/8e3f1695e058837ed80fec3238ff439fdc2d0f0e
while true; do
  if [[ -f /root/logs/r931_train.done ]]; then break; fi
  pid=$(cat /root/logs/r931_train.pid 2>/dev/null || true)
  if [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] && ! kill -0 "$pid" 2>/dev/null; then
    if [[ -d /root/r931/train ]] && ls /root/r931/train/adapter_model.safetensors >/dev/null 2>&1; then
      echo "[r931-wait] train exited with adapter; marking done"
      date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r931_train.done
      break
    fi
    echo "[r931-wait] FATAL train dead without adapter"; exit 1
  fi
  sleep 30
done
echo "[r931-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) MERGE start"
ADAPTER=/root/r931/train
if [[ -f /root/r931/train/adapter/adapter_config.json ]]; then ADAPTER=/root/r931/train/adapter; fi
mkdir -p /tmp/r931_merged
python3 /root/mining_src/s4-h1-sft/merge_lora.py \
  --base "$BASE" --adapter "$ADAPTER" --out /tmp/r931_merged \
  >/root/logs/r931_merge.nohup 2>&1
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r931_merge.done
echo "[r931-wait] MERGE done $(date -u +%Y-%m-%dT%H:%M:%SZ) adapter=$ADAPTER"
