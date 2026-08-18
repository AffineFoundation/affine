#!/usr/bin/env bash
# p3851: wait R796 MERGE_DONE → chall+v4-n80 on zesty GPUs 4,5. Never pkill -f.
set -euo pipefail
log() { echo "[p3851-r796-wait-n80] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
MERGE_DONE=/root/logs/r796_merge.done
MERGE_DIR=/tmp/r796_merged
LAUNCHED=/root/logs/r796_n80_launched.p3851
N80_SCRIPT=/root/mining_src/r796-marsplan-offline-dpo-hialpha-midrank-midbeta-midctx-megasuperextrasteps-ep4-ultralolr/lean_chall_n80_zesty_gpus45_p3851.sh
mkdir -p /root/logs
[[ -f "$LAUNCHED" ]] && { log "already launched"; exit 0; }
log "armed wait R796 MERGE → chall+n80 GPUs 4,5"
while true; do
  n=$(ls "$MERGE_DIR"/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
  if [[ -f "$MERGE_DONE" && -f "$MERGE_DIR/config.json" && "${n:-0}" -ge 16 ]]; then
    if pgrep -af "merge_lora.py.*r796" >/dev/null 2>&1; then
      log "merge.done present but merge_lora still running; wait"
      sleep 15
      continue
    fi
    log "MERGE_DONE shards=$n — launch chall+n80"
    date -u +%Y-%m-%dT%H:%M:%SZ >"$LAUNCHED"
    nohup bash "$N80_SCRIPT" >/root/logs/p3851_r796_lean.outer.nohup 2>&1 &
    echo $! >/root/logs/p3851_r796_lean.outer.pid
    log "n80 outer pid=$(cat /root/logs/p3851_r796_lean.outer.pid)"
    exit 0
  fi
  log "waiting merge_done=$([[ -f $MERGE_DONE ]] && echo y || echo n) shards=${n:-0}"
  sleep 20
done
