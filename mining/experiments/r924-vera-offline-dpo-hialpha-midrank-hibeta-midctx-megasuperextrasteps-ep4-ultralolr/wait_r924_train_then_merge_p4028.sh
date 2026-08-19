#!/usr/bin/env bash
set -euo pipefail
exec >>/root/logs/r924_wait_merge.nohup 2>&1
echo "[r924-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) wait train.done"
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
BASE=/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/8e3f1695e058837ed80fec3238ff439fdc2d0f0e
while true; do
  if [[ -f /root/logs/r924_train.done ]]; then break; fi
  pid=$(cat /root/logs/r924_train.pid 2>/dev/null || true)
  if [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] && ! kill -0 "$pid" 2>/dev/null; then
    if [[ -d /root/r924/train ]] && ls /root/r924/train/adapter_model.safetensors >/dev/null 2>&1; then
      echo "[r924-wait] train exited with adapter; marking done"
      date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r924_train.done
      break
    fi
    echo "[r924-wait] FATAL train dead without adapter"; exit 1
  fi
  sleep 30
done
echo "[r924-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) MERGE start"
mkdir -p /tmp/r924_merged
python3 /root/mining_src/s4-h1-sft/merge_lora.py \
  --base "$BASE" --adapter /root/r924/train --out /tmp/r924_merged \
  >/root/logs/r924_merge.nohup 2>&1
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r924_merge.done
echo "[r924-wait] MERGE done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
