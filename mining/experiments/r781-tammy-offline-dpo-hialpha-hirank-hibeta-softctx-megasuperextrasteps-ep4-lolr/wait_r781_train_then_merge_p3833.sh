#!/usr/bin/env bash
# p3833: wait R781 TRAIN_DONE → merge GPUs 2,3 — never pkill -f. No local n80.
set -euo pipefail
log() { echo "[p3833-r781-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
TRAIN_PID_FILE=/root/logs/r781_train.pid
ADAPTER=/root/r781/train/adapter
MERGE_SCRIPT=/root/mining_src/r781-tammy-offline-dpo-hialpha-hirank-hibeta-softctx-megasuperextrasteps-ep4-lolr/lean_merge_brave_gpus23_p3833.sh
LAUNCHED=/root/logs/r781_merge_launched.p3833
mkdir -p /root/logs
[[ -f "$LAUNCHED" ]] && { log "already launched"; exit 0; }
log "armed wait R781 → merge GPUs 2,3"
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
    date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r781_train.done
    nohup bash "$MERGE_SCRIPT" >/root/logs/p3833_r781_merge.outer.nohup 2>&1 &
    echo $! >/root/logs/p3833_r781_merge.outer.pid
    log "merge outer pid=$(cat /root/logs/p3833_r781_merge.outer.pid)"
    exit 0
  fi
  step=$(grep -o '"step": [0-9]*' /root/logs/r781_train.nohup 2>/dev/null | tail -1 | awk '{print $2}' || true)
  log "waiting train_alive=$train_alive adapter_ok=$adapter_ok step=${step:-?}"
  sleep 30
done
