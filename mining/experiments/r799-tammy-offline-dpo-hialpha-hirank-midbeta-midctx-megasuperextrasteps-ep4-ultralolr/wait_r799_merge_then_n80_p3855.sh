#!/usr/bin/env bash
# p3855: wait R799 MERGE_DONE → chall+v4-n80 on crown GPUs 6,7. Never pkill -f.
set -euo pipefail
log() { echo "[p3855-r799-wait-n80] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
MERGE_DONE=/root/logs/r799_merge.done
MERGE_DIR=/tmp/r799_merged
LAUNCHED=/root/logs/r799_n80_launched.p3855
N80_SCRIPT=/root/mining_src/r799-tammy-offline-dpo-hialpha-hirank-midbeta-midctx-megasuperextrasteps-ep4-ultralolr/lean_chall_n80_crown_gpus67_p3855.sh
mkdir -p /root/logs
[[ -f "$LAUNCHED" ]] && { log "already launched"; exit 0; }
log "armed wait R799 MERGE → chall+n80 GPUs 6,7"
while true; do
  n=$(ls "$MERGE_DIR"/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
  if [[ -f "$MERGE_DONE" && -f "$MERGE_DIR/config.json" && "${n:-0}" -ge 16 ]]; then
    if pgrep -af "merge_lora.py.*r799" >/dev/null 2>&1; then
      log "merge.done present but merge_lora still running; wait"
      sleep 15
      continue
    fi
    log "MERGE_DONE shards=$n — launch chall+n80"
    date -u +%Y-%m-%dT%H:%M:%SZ >"$LAUNCHED"
    nohup bash "$N80_SCRIPT" >/root/logs/p3855_r799_lean.outer.nohup 2>&1 &
    echo $! >/root/logs/p3855_r799_lean.outer.pid
    log "n80 outer pid=$(cat /root/logs/p3855_r799_lean.outer.pid)"
    exit 0
  fi
  log "waiting merge_done=$([[ -f $MERGE_DONE ]] && echo y || echo n) shards=${n:-0}"
  sleep 20
done
