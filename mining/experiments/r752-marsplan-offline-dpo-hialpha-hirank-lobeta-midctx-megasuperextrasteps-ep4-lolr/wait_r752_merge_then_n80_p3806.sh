#!/usr/bin/env bash
# p3806: wait R752 MERGE_DONE → chall+v4-n80 on lunar GPUs 4,5. Never pkill -f.
set -euo pipefail
log() { echo "[p3806-r752-wait-n80] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
MERGE_DONE=/root/logs/r752_merge.done
MERGE_DIR=/tmp/r752_merged
LAUNCHED=/root/logs/r752_n80_launched.p3806
N80_SCRIPT=/root/mining_src/r752-marsplan-offline-dpo-hialpha-hirank-lobeta-midctx-megasuperextrasteps-ep4-lolr/lean_chall_n80_lunar_gpus45_p3806.sh
mkdir -p /root/logs
[[ -f "$LAUNCHED" ]] && { log "already launched"; exit 0; }
log "armed wait R752 MERGE → chall+n80 GPUs 4,5"
while true; do
  n=$(ls "$MERGE_DIR"/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
  if [[ -f "$MERGE_DONE" && -f "$MERGE_DIR/config.json" && "${n:-0}" -ge 16 ]]; then
    if pgrep -af "merge_lora.py.*r752" >/dev/null 2>&1; then
      log "merge.done present but merge_lora still running; wait"
      sleep 15
      continue
    fi
    log "MERGE_DONE shards=$n — launch chall+n80"
    date -u +%Y-%m-%dT%H:%M:%SZ >"$LAUNCHED"
    nohup bash "$N80_SCRIPT" >/root/logs/p3806_r752_lean.outer.nohup 2>&1 &
    echo $! >/root/logs/p3806_r752_lean.outer.pid
    log "n80 outer pid=$(cat /root/logs/p3806_r752_lean.outer.pid)"
    exit 0
  fi
  log "waiting merge_done=$([[ -f $MERGE_DONE ]] && echo y || echo n) shards=${n:-0}"
  sleep 20
done
