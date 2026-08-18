#!/usr/bin/env bash
# p3792: wait R741 MERGE_DONE → lean chall+v4 n80 on lunar GPUs 4,5. Never pkill -f. Leave R749 on 6,7.
set -euo pipefail
log() { echo "[p3792-r741-wait-n80] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
MERGE_DONE=/root/logs/r741_merge.done
MERGE_DIR=/tmp/r741_merged
LEAN=/root/mining_src/r741-marsplan-offline-dpo-hialpha-hirank-lobeta-midctx-hyperextrasteps-ep3-lolr/lean_chall_n80_lunar_gpus45_p3792.sh
LAUNCHED=/root/logs/r741_n80_launched.p3792
mkdir -p /root/logs
[[ -f "$LAUNCHED" ]] && { log "already launched"; exit 0; }
log "armed wait R741 MERGE → chall/:8002 + v4 n80 GPUs 4,5"
hub_ok() {
  local n; n=$(ls "$1"/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
  [[ -f "$1/config.json" && "${n:-0}" -ge 16 ]]
}
while true; do
  if [[ -f "$MERGE_DONE" ]] && hub_ok "$MERGE_DIR"; then
    if pgrep -f "merge_lora.py .*r741" >/dev/null 2>&1; then
      log "merge.done present but merge_lora still live — wait"
      sleep 15; continue
    fi
    log "MERGE_DONE — launch lean chall+n80"
    date -u +%Y-%m-%dT%H:%M:%SZ >"$LAUNCHED"
    nohup bash "$LEAN" >/root/logs/p3792_r741_lean.outer.log 2>&1 &
    echo $! >/root/logs/p3792_r741_lean.outer.pid
    log "lean outer pid=$(cat /root/logs/p3792_r741_lean.outer.pid)"
    exit 0
  fi
  n=$(ls "$MERGE_DIR"/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
  log "waiting merge_done=$([[ -f $MERGE_DONE ]] && echo 1 || echo 0) shards=${n:-0}"
  sleep 20
done
