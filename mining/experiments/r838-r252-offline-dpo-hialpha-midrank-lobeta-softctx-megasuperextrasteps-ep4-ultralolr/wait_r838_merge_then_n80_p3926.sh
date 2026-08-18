#!/usr/bin/env bash
set -euo pipefail
log() { echo "[p3926-r838-n80-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
MERGE_DONE=/root/logs/r838_merge.done
CHALL=/root/mining_src/r838-r252-offline-dpo-hialpha-midrank-lobeta-softctx-megasuperextrasteps-ep4-ultralolr/lean_chall_n80_golden_gpus45_p3926.sh
LAUNCHED=/root/logs/r838_n80_launched.p3926
mkdir -p /root/logs
[[ -f "$LAUNCHED" ]] && { log "already launched"; exit 0; }
log "armed wait R838 MERGE → n80 GPUs 4,5"
while true; do
  if [[ -f "$MERGE_DONE" ]] && [[ -f /tmp/r838_merged/config.json ]]; then
    n=$(ls /tmp/r838_merged/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
    if [[ "${n:-0}" -ge 16 ]]; then
      log "MERGE_READY shards=$n — launch chall+n80"
      date -u +%Y-%m-%dT%H:%M:%SZ >"$LAUNCHED"
      nohup bash "$CHALL" >/root/logs/p3926_r838_chall_n80_wvk7.log 2>&1 &
      echo $! >/root/logs/p3926_r838_chall_n80.pid
      log "chall pid=$(cat /root/logs/p3926_r838_chall_n80.pid)"
      exit 0
    fi
  fi
  log "waiting merge_done=$([[ -f $MERGE_DONE ]] && echo 1 || echo 0)"
  sleep 30
done
