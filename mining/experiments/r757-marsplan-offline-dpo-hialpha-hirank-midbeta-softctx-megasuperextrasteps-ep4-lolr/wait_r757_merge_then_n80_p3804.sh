#!/usr/bin/env bash
# p3804: wait R757 MERGE_DONE → chall+v4-n80 on zesty GPUs 4,5. Never pkill -f.
set -euo pipefail
log() { echo "[p3804-r757-wait-n80] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
MERGE_DONE=/root/logs/r757_merge.done
MERGE_DIR=/tmp/r757_merged
LAUNCHED=/root/logs/r757_n80_launched.p3804
N80_SCRIPT=/root/mining_src/r757-marsplan-offline-dpo-hialpha-hirank-midbeta-softctx-megasuperextrasteps-ep4-lolr/lean_chall_n80_zesty_gpus45_p3804.sh
mkdir -p /root/logs
[[ -f "$LAUNCHED" ]] && { log "already launched"; exit 0; }
log "armed wait R757 MERGE → chall+n80 GPUs 4,5"
while true; do
  n=$(ls "$MERGE_DIR"/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
  if [[ -f "$MERGE_DONE" && -f "$MERGE_DIR/config.json" && "${n:-0}" -ge 16 ]]; then
    if pgrep -af "merge_lora.py.*r757" >/dev/null 2>&1; then
      log "merge.done present but merge_lora still running; wait"
      sleep 15
      continue
    fi
    log "MERGE_DONE shards=$n — launch chall+n80"
    date -u +%Y-%m-%dT%H:%M:%SZ >"$LAUNCHED"
    nohup bash "$N80_SCRIPT" >/root/logs/p3804_r757_lean.outer.nohup 2>&1 &
    echo $! >/root/logs/p3804_r757_lean.outer.pid
    log "n80 outer pid=$(cat /root/logs/p3804_r757_lean.outer.pid)"
    exit 0
  fi
  log "waiting merge_done=$([[ -f $MERGE_DONE ]] && echo y || echo n) shards=${n:-0}"
  sleep 20
done
