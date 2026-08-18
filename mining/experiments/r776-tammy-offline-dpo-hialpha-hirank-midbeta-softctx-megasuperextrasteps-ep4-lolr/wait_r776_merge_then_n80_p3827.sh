#!/usr/bin/env bash
# p3827: wait R776 MERGE_DONE → chall+v4-n80 on crown GPUs 6,7. Never pkill -f.
set -euo pipefail
log() { echo "[p3827-r776-wait-n80] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
MERGE_DONE=/root/logs/r776_merge.done
MERGE_DIR=/tmp/r776_merged
LAUNCHED=/root/logs/r776_n80_launched.p3827
N80_SCRIPT=/root/mining_src/r776-tammy-offline-dpo-hialpha-hirank-midbeta-softctx-megasuperextrasteps-ep4-lolr/lean_chall_n80_crown_gpus67_p3827.sh
mkdir -p /root/logs
[[ -f "$LAUNCHED" ]] && { log "already launched"; exit 0; }
log "armed wait R776 MERGE → chall+n80 GPUs 6,7"
while true; do
  n=$(ls "$MERGE_DIR"/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
  if [[ -f "$MERGE_DONE" && -f "$MERGE_DIR/config.json" && "${n:-0}" -ge 16 ]]; then
    if pgrep -af "merge_lora.py.*r776" >/dev/null 2>&1; then
      log "merge.done present but merge_lora still running; wait"
      sleep 15
      continue
    fi
    log "MERGE_DONE shards=$n — launch chall+n80"
    date -u +%Y-%m-%dT%H:%M:%SZ >"$LAUNCHED"
    nohup bash "$N80_SCRIPT" >/root/logs/p3827_r776_lean.outer.nohup 2>&1 &
    echo $! >/root/logs/p3827_r776_lean.outer.pid
    log "n80 outer pid=$(cat /root/logs/p3827_r776_lean.outer.pid)"
    exit 0
  fi
  log "waiting merge_done=$([[ -f $MERGE_DONE ]] && echo y || echo n) shards=${n:-0}"
  sleep 20
done
