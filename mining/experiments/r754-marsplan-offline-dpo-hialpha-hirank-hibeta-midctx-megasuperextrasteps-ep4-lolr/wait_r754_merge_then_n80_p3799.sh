#!/usr/bin/env bash
set -euo pipefail
log() { echo "[p3799-r754-wait-n80] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
MERGE_DONE=/root/logs/r754_merge.done
MERGE_DIR=/tmp/r754_merged
LAUNCHED=/root/logs/r754_n80_launched.p3799
N80_SCRIPT=/root/mining_src/r754-marsplan-offline-dpo-hialpha-hirank-hibeta-midctx-megasuperextrasteps-ep4-lolr/lean_chall_n80_zesty_gpus67_p3799.sh
mkdir -p /root/logs
[[ -f "$LAUNCHED" ]] && { log "already launched"; exit 0; }
log "armed wait R754 MERGE -> chall+n80 GPUs 6,7"
while true; do
  n=$(ls "$MERGE_DIR"/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
  if [[ -f "$MERGE_DONE" && -f "$MERGE_DIR/config.json" && "${n:-0}" -ge 16 ]]; then
    if pgrep -af "merge_lora.py.*r754" >/dev/null 2>&1; then
      log "merge.done present but merge_lora still running; wait"
      sleep 15
      continue
    fi
    log "MERGE_DONE shards=$n — launch chall+n80"
    date -u +%Y-%m-%dT%H:%M:%SZ >"$LAUNCHED"
    nohup bash "$N80_SCRIPT" >/root/logs/p3799_r754_lean.outer.nohup 2>&1 &
    echo $! >/root/logs/p3799_r754_lean.outer.pid
    log "n80 outer pid=$(cat /root/logs/p3799_r754_lean.outer.pid)"
    exit 0
  fi
  log "waiting merge_done=$([[ -f $MERGE_DONE ]] && echo y || echo n) shards=${n:-0}"
  sleep 20
done
