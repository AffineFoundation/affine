#!/usr/bin/env bash
set -euo pipefail
log() { echo "[p3784-r746-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
TRAIN_PID_FILE=/root/logs/r746_train.pid
ADAPTER=/root/r746/train/adapter
MERGE_SCRIPT=/root/mining_src/r746-r252-offline-dpo-hialpha-hirank-lobeta-shortctx-superextrasteps-ep3-lolr/lean_merge_golden_gpus67_p3770.sh
LAUNCHED=/root/logs/r746_merge_launched.p3784
mkdir -p /root/logs
[[ -f "$LAUNCHED" ]] && { log already; exit 0; }
log armed
while true; do
  train_alive=0
  if [[ -f "$TRAIN_PID_FILE" ]]; then
    tpid=$(cat "$TRAIN_PID_FILE" 2>/dev/null || true)
    if [[ "$tpid" =~ ^[0-9]+$ ]] && kill -0 "$tpid" 2>/dev/null; then train_alive=1; fi
  fi
  adapter_ok=0
  [[ -f "$ADAPTER/adapter_config.json" && -f "$ADAPTER/adapter_model.safetensors" ]] && adapter_ok=1
  if [[ "$train_alive" -eq 0 && "$adapter_ok" -eq 1 ]]; then
    log TRAIN_DONE
    date -u +%Y-%m-%dT%H:%M:%SZ >"$LAUNCHED"
    date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r746_train.done
    nohup bash "$MERGE_SCRIPT" >/root/logs/p3784_r746_merge.outer.nohup 2>&1 &
    echo $! >/root/logs/p3784_r746_merge.outer.pid
    exit 0
  fi
  sleep 30
done
