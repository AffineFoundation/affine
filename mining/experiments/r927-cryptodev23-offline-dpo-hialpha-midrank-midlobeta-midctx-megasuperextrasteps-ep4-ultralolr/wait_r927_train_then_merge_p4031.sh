#!/usr/bin/env bash
set -euo pipefail
exec >>/root/logs/r927_wait_merge.nohup 2>&1
echo "[r927-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) wait train.done"
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
BASE=/root/hf/hub/models--cryptoDev23--Affine-5Dku3dYp9j-hk8161/snapshots/55b7ffe003d078a8a131673f677b2584548a502e
while true; do
  if [[ -f /root/logs/r927_train.done ]]; then break; fi
  pid=$(cat /root/logs/r927_train.pid 2>/dev/null || true)
  if [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] && ! kill -0 "$pid" 2>/dev/null; then
    if [[ -d /root/r927/train ]] && test -f /root/r927/train/adapter/adapter_model.safetensors >/dev/null 2>&1; then
      echo "[r927-wait] train exited with adapter; marking done"
      date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r927_train.done
      break
    fi
    echo "[r927-wait] FATAL train dead without adapter"; exit 1
  fi
  # progress heartbeat
  step=$(grep -oE '"step": [0-9]+' /root/logs/r927_train.nohup 2>/dev/null | tail -1 | awk '{print $2}' || echo 0)
  adapter_ok=0
  [[ -f /root/r927/train/adapter/adapter_model.safetensors ]] && adapter_ok=1
  alive=0
  [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null && alive=1
  echo "[r927-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) waiting train_alive=$alive adapter_ok=$adapter_ok step=${step:-0}"
  sleep 30
done
echo "[r927-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) MERGE start"
mkdir -p /tmp/r927_merged
python3 /root/mining_src/s4-h1-sft/merge_lora.py \
  --base "$BASE" --adapter /root/r927/train/adapter --out /tmp/r927_merged \
  >/root/logs/r927_merge.nohup 2>&1
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r927_merge.done
echo "[r927-wait] MERGE done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
