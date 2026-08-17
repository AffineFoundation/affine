#!/usr/bin/env bash
# p3681: wait R664 TRAIN_DONE → merge on GPUs 2,3
# Detect by PID file / adapter presence only — never pkill -f.
set -euo pipefail
log() { echo "[p3681-r664-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
TRAIN_PID_FILE=/root/logs/r664_train.pid
ADAPTER=/root/r664/train/adapter
MERGE_SCRIPT=/root/mining_src/r664-r252-offline-dpo-hialpha-midrank-midbeta-longctx-megaextrasteps-ep3-lolr/lean_merge_crown_gpus23_p3681.sh
LAUNCHED=/root/logs/r664_merge_launched.p3681
mkdir -p /root/logs
[[ -f "$LAUNCHED" ]] && { log "already launched"; exit 0; }
log "armed wait R664 → merge GPUs 2,3"
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
    date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r664_train.done
    nohup bash "$MERGE_SCRIPT" >/root/logs/p3681_r664_merge.outer.nohup 2>&1 &
    echo $! >/root/logs/p3681_r664_merge.outer.pid
    log "merge outer pid=$(cat /root/logs/p3681_r664_merge.outer.pid)"
    exit 0
  fi
  step=$(grep -o '"step": [0-9]*' /root/logs/r664_train.nohup 2>/dev/null | tail -1 | awk '{print $2}' || true)
  log "waiting train_alive=$train_alive adapter_ok=$adapter_ok step=${step:-?}"
  sleep 30
done
