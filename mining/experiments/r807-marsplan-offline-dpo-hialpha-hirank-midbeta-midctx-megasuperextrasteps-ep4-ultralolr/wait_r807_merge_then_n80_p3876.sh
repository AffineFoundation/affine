#!/usr/bin/env bash
set -euo pipefail
log() { echo "[p3876-r807-n80-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
MERGE_DONE=/root/logs/r807_merge.done
CHALL=/root/mining_src/r807-marsplan-offline-dpo-hialpha-hirank-midbeta-midctx-megasuperextrasteps-ep4-ultralolr/lean_chall_n80_lunar_gpus45_p3876.sh
LAUNCHED=/root/logs/r807_n80_launched.p3876
mkdir -p /root/logs
[[ -f "$LAUNCHED" ]] && { log "already launched"; exit 0; }
log "armed wait R807 MERGE → n80 GPUs 4,5"
while true; do
  if [[ -f "$MERGE_DONE" ]] && [[ -f /tmp/r807_merged/config.json ]]; then
    n=$(ls /tmp/r807_merged/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
    if [[ "${n:-0}" -ge 16 ]]; then
      log "MERGE_READY shards=$n — launch chall+n80"
      date -u +%Y-%m-%dT%H:%M:%SZ >"$LAUNCHED"
      nohup bash "$CHALL" >/root/logs/p3876_r807_chall_n80_wvk7.log 2>&1 &
      echo $! >/root/logs/p3876_r807_chall_n80.pid
      log "chall pid=$(cat /root/logs/p3876_r807_chall_n80.pid)"
      exit 0
    fi
  fi
  log "waiting merge_done=$([[ -f $MERGE_DONE ]] && echo 1 || echo 0)"
  sleep 30
done
