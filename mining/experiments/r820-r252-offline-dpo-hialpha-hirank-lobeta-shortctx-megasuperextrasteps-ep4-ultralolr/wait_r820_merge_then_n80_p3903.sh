#!/usr/bin/env bash
set -euo pipefail
log() { echo "[p3903-r820-n80-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
MERGE_DONE=/root/logs/r820_merge.done
CHALL=/root/mining_src/r820-r252-offline-dpo-hialpha-hirank-lobeta-shortctx-megasuperextrasteps-ep4-ultralolr/lean_chall_n80_r252_gpus45_p3903.sh
LAUNCHED=/root/logs/r820_n80_launched.p3903
mkdir -p /root/logs
[[ -f "$LAUNCHED" ]] && { log "already launched"; exit 0; }
log "armed wait R820 MERGE → n80 GPUs 4,5"
while true; do
  if [[ -f "$MERGE_DONE" ]] && [[ -f /tmp/r820_merged/config.json ]]; then
    n=$(ls /tmp/r820_merged/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
    if [[ "${n:-0}" -ge 16 ]]; then
      log "MERGE_READY shards=$n — launch chall+n80"
      date -u +%Y-%m-%dT%H:%M:%SZ >"$LAUNCHED"
      nohup bash "$CHALL" >/root/logs/p3903_r820_chall_n80_wvk7.log 2>&1 &
      echo $! >/root/logs/p3903_r820_chall_n80.pid
      log "chall pid=$(cat /root/logs/p3903_r820_chall_n80.pid)"
      exit 0
    fi
  fi
  log "waiting merge_done=$([[ -f $MERGE_DONE ]] && echo 1 || echo 0)"
  sleep 30
done
