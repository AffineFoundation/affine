#!/usr/bin/env bash
# p3801: wait R756 MERGE_DONE → chall+v4-n80 on golden GPUs 6,7. Never pkill -f.
set -euo pipefail
log() { echo "[p3801-r756-wait-n80] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
MERGE_DONE=/root/logs/r756_merge.done
MERGE_DIR=/tmp/r756_merged
LAUNCHED=/root/logs/r756_n80_launched.p3801
N80_SCRIPT=/root/mining_src/r756-r252-offline-dpo-hialpha-hirank-lobeta-shortctx-megasuperextrasteps-ep4-lolr/lean_chall_n80_golden_gpus67_p3801.sh
mkdir -p /root/logs
[[ -f "$LAUNCHED" ]] && { log "already launched"; exit 0; }
log "armed wait R756 MERGE → chall+n80 GPUs 6,7"
while true; do
  n=$(ls "$MERGE_DIR"/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
  if [[ -f "$MERGE_DONE" && -f "$MERGE_DIR/config.json" && "${n:-0}" -ge 16 ]]; then
    if pgrep -af "merge_lora.py.*r756" >/dev/null 2>&1; then
      log "merge.done present but merge_lora still running; wait"
      sleep 15
      continue
    fi
    log "MERGE_DONE shards=$n — launch chall+n80"
    date -u +%Y-%m-%dT%H:%M:%SZ >"$LAUNCHED"
    nohup bash "$N80_SCRIPT" >/root/logs/p3801_r756_lean.outer.nohup 2>&1 &
    echo $! >/root/logs/p3801_r756_lean.outer.pid
    log "n80 outer pid=$(cat /root/logs/p3801_r756_lean.outer.pid)"
    exit 0
  fi
  log "waiting merge_done=$([[ -f $MERGE_DONE ]] && echo y || echo n) shards=${n:-0}"
  sleep 20
done
