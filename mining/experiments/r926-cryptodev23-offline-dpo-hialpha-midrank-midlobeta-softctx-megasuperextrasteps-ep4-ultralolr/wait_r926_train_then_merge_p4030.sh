#!/usr/bin/env bash
set -euo pipefail
exec >>/root/logs/r926_wait_merge.nohup 2>&1
echo "[r926-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) wait train.done"
set -a; source /root/mine.env; set +a
source /root/venv/bin/activate
export HF_HOME=/root/hf PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
BASE=/root/hf/hub/models--cryptoDev23--Affine-5Dku3dYp9j-hk8161/snapshots/55b7ffe003d078a8a131673f677b2584548a502e
while true; do
  if [[ -f /root/logs/r926_train.done ]]; then break; fi
  pid=$(cat /root/logs/r926_train.pid 2>/dev/null || true)
  if [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] && ! kill -0 "$pid" 2>/dev/null; then
    if [[ -d /root/r926/train ]] && ls /root/r926/train/adapter_model.safetensors >/dev/null 2>&1; then
      echo "[r926-wait] train exited with adapter; marking done"
      date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r926_train.done
      break
    fi
    echo "[r926-wait] FATAL train dead without adapter"; exit 1
  fi
  sleep 30
done
echo "[r926-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) MERGE start"
mkdir -p /tmp/r926_merged
python3 /root/mining_src/s4-h1-sft/merge_lora.py \
  --base "$BASE" --adapter /root/r926/train --out /tmp/r926_merged \
  >/root/logs/r926_merge.nohup 2>&1
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r926_merge.done
echo "[r926-wait] MERGE done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
