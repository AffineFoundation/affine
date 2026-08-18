#!/usr/bin/env bash
# p3860: wait R803 MERGE_DONE → chall+v4-n80 on golden GPUs 6,7. Never pkill -f.
set -euo pipefail
log() { echo "[p3860-r803-wait-n80] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
MERGE_DONE=/root/logs/r803_merge.done
MERGE_DIR=/tmp/r803_merged
LAUNCHED=/root/logs/r803_n80_launched.p3860
N80_SCRIPT=/root/mining_src/r803-r252-offline-dpo-hialpha-midrank-hibeta-midctx-megasuperextrasteps-ep4-ultralolr/lean_chall_n80_golden_gpus67_p3860.sh
mkdir -p /root/logs
[[ -f "$LAUNCHED" ]] && { log "already launched"; exit 0; }
log "armed wait R803 MERGE → chall+n80 GPUs 6,7"
while true; do
  n=$(ls "$MERGE_DIR"/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
  if [[ -f "$MERGE_DONE" && -f "$MERGE_DIR/config.json" && "${n:-0}" -ge 16 ]]; then
    if pgrep -af "merge_lora.py.*r803" >/dev/null 2>&1; then
      log "merge.done present but merge_lora still running; wait"
      sleep 15
      continue
    fi
    log "MERGE_DONE shards=$n — launch chall+n80"
    date -u +%Y-%m-%dT%H:%M:%SZ >"$LAUNCHED"
    nohup bash "$N80_SCRIPT" >/root/logs/p3860_r803_lean.outer.nohup 2>&1 &
    echo $! >/root/logs/p3860_r803_lean.outer.pid
    log "n80 outer pid=$(cat /root/logs/p3860_r803_lean.outer.pid)"
    exit 0
  fi
  log "waiting merge_done=$([[ -f $MERGE_DONE ]] && echo y || echo n) shards=${n:-0}"
  sleep 20
done
