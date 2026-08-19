#!/usr/bin/env bash
set -euo pipefail
exec >>/root/logs/r937_wait_merge.nohup 2>&1
echo "[r937-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) wait train.done"
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
BASE=/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/8e3f1695e058837ed80fec3238ff439fdc2d0f0e
while true; do
  if [[ -f /root/logs/r937_train.done || -f /root/r937/train/train.done ]]; then
    [[ -f /root/logs/r937_train.done ]] || cp -f /root/r937/train/train.done /root/logs/r937_train.done
    break
  fi
  pid=$(cat /root/logs/r937_train.pid 2>/dev/null || true)
  if [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] && ! kill -0 "$pid" 2>/dev/null; then
    if [[ -f /root/r937/train/adapter_model.safetensors ]]; then
      echo "[r937-wait] train exited with adapter; marking done"
      date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r937_train.done
      break
    fi
    if [[ -f /root/r937/train/adapter/adapter_model.safetensors ]]; then
      echo "[r937-wait] train exited with adapter/; marking done"
      date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r937_train.done
      break
    fi
    echo "[r937-wait] FATAL train dead without adapter"; exit 1
  fi
  step=$(grep -oE '"step": [0-9]+' /root/logs/r937_train.nohup 2>/dev/null | tail -1 | awk '{print $2}' || echo 0)
  adapter_ok=0
  [[ -f /root/r937/train/adapter_model.safetensors || -f /root/r937/train/adapter/adapter_model.safetensors ]] && adapter_ok=1
  alive=0
  [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null && alive=1
  echo "[r937-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) waiting train_alive=$alive adapter_ok=$adapter_ok step=${step:-0}"
  sleep 30
done
echo "[r937-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) MERGE start"
ADAPTER=/root/r937/train
[[ -f /root/r937/train/adapter/adapter_model.safetensors ]] && ADAPTER=/root/r937/train/adapter
mkdir -p /tmp/r937_merged
python3 /root/mining_src/s4-h1-sft/merge_lora.py \
  --base "$BASE" --adapter "$ADAPTER" --out /tmp/r937_merged \
  >/root/logs/r937_merge.nohup 2>&1
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r937_merge.done
echo "[r937-wait] MERGE done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
