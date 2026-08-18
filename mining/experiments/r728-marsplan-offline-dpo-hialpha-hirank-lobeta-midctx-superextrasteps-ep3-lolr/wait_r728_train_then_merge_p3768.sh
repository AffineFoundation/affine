#!/usr/bin/env bash
# p3768: wait R728 train → merge on lunar GPUs 6,7
set -euo pipefail
LOG=/root/logs/p3768_r728_wait.nohup
exec >>"$LOG" 2>&1
echo "[p3768-r728-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) start"
while true; do
  alive=0
  if [[ -f /root/logs/r728_train.pid ]]; then
    p=$(cat /root/logs/r728_train.pid 2>/dev/null || true)
    if [[ -n "${p:-}" && "$p" =~ ^[0-9]+$ ]] && kill -0 "$p" 2>/dev/null; then alive=1; fi
  fi
  adapter_ok=0
  [[ -f /root/r728/train/adapter/adapter_model.safetensors || -f /root/r728/train/adapter/adapter_model.bin ]] && adapter_ok=1
  step=$(grep -o '"step": [0-9]*' /root/logs/r728_train.nohup 2>/dev/null | tail -1 | awk '{print $2}' || true)
  echo "[p3768-r728-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) waiting train_alive=$alive adapter_ok=$adapter_ok step=${step:-?}"
  if [[ "$alive" -eq 0 && "$adapter_ok" -eq 1 ]]; then
    date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r728_train.done
    echo "[p3768-r728-wait] TRAIN_DONE — launch merge"
    break
  fi
  if [[ "$alive" -eq 0 && "$adapter_ok" -eq 0 ]]; then
    echo "FATAL train dead without adapter"; exit 1
  fi
  sleep 30
done
# reuse r721 merge lean adapted
MERGE_SH=/root/mining_src/r728-marsplan-offline-dpo-hialpha-hirank-lobeta-midctx-superextrasteps-ep3-lolr/lean_merge_lunar_gpus67_p3768.sh
if [[ -x "$MERGE_SH" ]]; then
  nohup bash "$MERGE_SH" > /root/logs/p3768_r728_merge.outer.nohup 2>&1 &
  echo $! > /root/logs/p3768_r728_merge.outer.pid
  echo "[p3768-r728-wait] merge outer pid=$(cat /root/logs/p3768_r728_merge.outer.pid)"
else
  echo "WARN no merge script yet — train done only"
fi
