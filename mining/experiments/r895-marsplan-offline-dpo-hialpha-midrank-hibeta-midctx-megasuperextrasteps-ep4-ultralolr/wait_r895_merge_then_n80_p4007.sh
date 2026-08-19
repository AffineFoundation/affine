#!/usr/bin/env bash
# p4007: wait R895 MERGE_DONE → chall+n80 on R337 GPUs 4,5 :8002. Never pkill -f.
set -euo pipefail
log() { echo "[p4007-r895-wait-n80] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
LAUNCHED=/root/logs/r895_n80_launched.p4007
MERGE_DONE=/root/logs/r895_merge.done
MERGE_DIR=/tmp/r895_merged
CHALL=/root/mining_src/r895-marsplan-offline-dpo-hialpha-midrank-hibeta-midctx-megasuperextrasteps-ep4-ultralolr/lean_chall_n80_r337_gpus45_p4007.sh
mkdir -p /root/logs
[[ -f "$LAUNCHED" ]] && { log "already launched"; exit 0; }
log "armed wait R895 merge → n80"
while true; do
  if [[ -f "$MERGE_DONE" ]] && [[ -f "$MERGE_DIR/config.json" ]]; then
    n=$(ls "$MERGE_DIR"/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
    if [[ "${n:-0}" -ge 16 ]]; then
      log "MERGE_DONE shards=$n — launch n80"
      date -u +%Y-%m-%dT%H:%M:%SZ >"$LAUNCHED"
      nohup bash "$CHALL" >/root/logs/p4007_r895_n80.outer.nohup 2>&1 &
      echo $! >/root/logs/p4007_r895_n80.outer.pid
      log "n80 outer pid=$(cat /root/logs/p4007_r895_n80.outer.pid)"
      exit 0
    fi
  fi
  log "waiting merge.done + shards (have_done=$([[ -f $MERGE_DONE ]] && echo y || echo n))"
  sleep 20
done
