#!/usr/bin/env bash
set -euo pipefail
log() { echo "[p4022-r921-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
TRAIN_PID_FILE=/root/logs/r921_train.pid
ADAPTER=/root/r921/train/adapter
MERGE_SCRIPT=/root/mining_src/r921-tammy-offline-dpo-hialpha-midrank-lobeta-softctx-megasuperextrasteps-ep4-ultralolr/lean_merge_brave_gpus45_p4022.sh
LAUNCHED=/root/logs/r921_merge_launched.p4022
mkdir -p /root/logs
[[ -f "$LAUNCHED" ]] && { log already_launched; exit 0; }
log "armed wait R921 → merge GPUs 4,5"
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
    date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r921_train.done
    nohup bash "$MERGE_SCRIPT" >/root/logs/p4022_r921_merge.outer.nohup 2>&1 &
    echo $! >/root/logs/p4022_r921_merge.outer.pid
    log "merge outer pid=$(cat /root/logs/p4022_r921_merge.outer.pid)"; exit 0
  fi
  step=$(grep -oE '"step": [0-9]+' /root/logs/r921_train.nohup 2>/dev/null | tail -1 | awk '{print $2}' || true)
  log "waiting train_alive=$train_alive adapter_ok=$adapter_ok step=${step:-?}"
  sleep 30
done
