#!/usr/bin/env bash
set -euo pipefail
log(){ echo "[p3955-r868-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
TRAIN_PID_FILE=/root/logs/r868_train.pid; ADAPTER=/root/r868/train/adapter
MERGE_SCRIPT=/root/mining_src/r868-r252-offline-dpo-hialpha-midrank-midbeta-softhihi-softctx-megasuperextrasteps-ep4-ultralolr/lean_merge_r252_gpus67_p3955.sh
LAUNCHED=/root/logs/r868_merge_launched.p3955; mkdir -p /root/logs
[[ -f "$LAUNCHED" ]] && { log already; exit 0; }
log "armed R868 -> merge 6,7"
while true; do
  train_alive=0
  if [[ -f "$TRAIN_PID_FILE" ]]; then tpid=$(cat "$TRAIN_PID_FILE" 2>/dev/null || true); [[ "$tpid" =~ ^[0-9]+$ ]] && kill -0 "$tpid" 2>/dev/null && train_alive=1; fi
  adapter_ok=0; [[ -f "$ADAPTER/adapter_config.json" && -f "$ADAPTER/adapter_model.safetensors" ]] && adapter_ok=1
  if [[ "$train_alive" -eq 0 && "$adapter_ok" -eq 1 ]]; then
    log TRAIN_DONE; date -u +%Y-%m-%dT%H:%M:%SZ >"$LAUNCHED"
    nohup bash "$MERGE_SCRIPT" >/root/logs/p3955_r868_merge.outer.nohup 2>&1 &
    echo $! >/root/logs/p3955_r868_merge.outer.pid; exit 0
  fi
  step=$(grep -oE '"step": [0-9]+' /root/logs/r868_train.nohup 2>/dev/null | tail -1 | awk '{print $2}' || true)
  log "wait alive=$train_alive adapter=$adapter_ok step=${step:-?}"; sleep 30
done
