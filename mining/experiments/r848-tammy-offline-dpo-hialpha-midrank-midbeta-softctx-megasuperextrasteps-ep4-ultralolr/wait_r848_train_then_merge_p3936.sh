#!/usr/bin/env bash
set -euo pipefail
log() { echo "[p3936-r848-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
TRAIN_PID_FILE=/root/logs/r848_train.pid
ADAPTER=/root/r848/train/adapter
MERGE_SCRIPT=/root/mining_src/r848-tammy-offline-dpo-hialpha-midrank-midbeta-softctx-megasuperextrasteps-ep4-ultralolr/lean_merge_brave_gpus01_p3936.sh
LAUNCHED=/root/logs/r848_merge_launched.p3936
mkdir -p /root/logs
[[ -f "$LAUNCHED" ]] && { log "already launched"; exit 0; }
log "armed wait R848 → merge GPUs 0,1"
while true; do
  train_alive=0
  if [[ -f "$TRAIN_PID_FILE" ]]; then
    tpid=$(cat "$TRAIN_PID_FILE" 2>/dev/null || true)
    if [[ "$tpid" =~ ^[0-9]+$ ]] && kill -0 "$tpid" 2>/dev/null; then train_alive=1; fi
  fi
  adapter_ok=0
  [[ -f "$ADAPTER/adapter_config.json" && -f "$ADAPTER/adapter_model.safetensors" ]] && adapter_ok=1
  if [[ "$train_alive" -eq 0 && "$adapter_ok" -eq 1 ]]; then
    log "TRAIN_DONE — launch merge"
    date -u +%Y-%m-%dT%H:%M:%SZ >"$LAUNCHED"
    date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r848_train.done
    nohup bash "$MERGE_SCRIPT" >/root/logs/p3936_r848_merge.outer.nohup 2>&1 &
    echo $! >/root/logs/p3936_r848_merge.outer.pid
    log "merge outer pid=$(cat /root/logs/p3936_r848_merge.outer.pid)"
    exit 0
  fi
  step=$(grep -oE '"step": [0-9]+' /root/logs/r848_train.nohup 2>/dev/null | tail -1 | awk '{print $2}' || true)
  log "waiting train_alive=$train_alive adapter_ok=$adapter_ok step=${step:-?}"
  sleep 30
done
