#!/usr/bin/env bash
# p3695: wait R675 TRAIN_DONE → merge on GPUs 6,7 — never pkill -f.
set -euo pipefail
log() { echo "[p3695-r675-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
TRAIN_PID_FILE=/root/logs/r675_train.pid
ADAPTER=/root/r675/train/adapter
MERGE_SCRIPT=/root/mining_src/r675-r252-offline-dpo-hialpha-midrank-midbeta-softctx-ultraextrasteps-ep3-lolr/lean_merge_r252_gpus67_p3695.sh
LAUNCHED=/root/logs/r675_merge_launched.p3695
mkdir -p /root/logs
[[ -f "$LAUNCHED" ]] && { log "already launched"; exit 0; }
log "armed wait R675 → merge GPUs 6,7"
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
    date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r675_train.done
    nohup bash "$MERGE_SCRIPT" >/root/logs/p3695_r675_merge.outer.nohup 2>&1 &
    echo $! >/root/logs/p3695_r675_merge.outer.pid
    log "merge outer pid=$(cat /root/logs/p3695_r675_merge.outer.pid)"
    exit 0
  fi
  step=$(grep -o '"step": [0-9]*' /root/logs/r675_train.nohup 2>/dev/null | tail -1 | awk '{print $2}' || true)
  log "waiting train_alive=$train_alive adapter_ok=$adapter_ok step=${step:-?}"
  sleep 30
done
