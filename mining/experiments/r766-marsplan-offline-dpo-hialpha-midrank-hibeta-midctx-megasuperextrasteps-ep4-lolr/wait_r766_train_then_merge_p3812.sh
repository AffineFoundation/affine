#!/usr/bin/env bash
# p3812: wait R766 TRAIN_DONE → merge on GPUs 6,7 — never pkill -f.
set -euo pipefail
log() { echo "[p3812-r766-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
TRAIN_PID_FILE=/root/logs/r766_train.pid
ADAPTER=/root/r766/train/adapter
MERGE_SCRIPT=/root/mining_src/r766-marsplan-offline-dpo-hialpha-midrank-hibeta-midctx-megasuperextrasteps-ep4-lolr/lean_merge_zesty_gpus67_p3812.sh
LAUNCHED=/root/logs/r766_merge_launched.p3812
mkdir -p /root/logs
[[ -f "$LAUNCHED" ]] && { log "already launched"; exit 0; }
log "armed wait R766 → merge GPUs 6,7"
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
    date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r766_train.done
    nohup bash "$MERGE_SCRIPT" >/root/logs/p3812_r766_merge.outer.nohup 2>&1 &
    echo $! >/root/logs/p3812_r766_merge.outer.pid
    log "merge outer pid=$(cat /root/logs/p3812_r766_merge.outer.pid)"
    exit 0
  fi
  step=$(grep -o '"step": [0-9]*' /root/logs/r766_train.nohup 2>/dev/null | tail -1 | awk '{print $2}' || true)
  log "waiting train_alive=$train_alive adapter_ok=$adapter_ok step=${step:-?}"
  sleep 30
done
