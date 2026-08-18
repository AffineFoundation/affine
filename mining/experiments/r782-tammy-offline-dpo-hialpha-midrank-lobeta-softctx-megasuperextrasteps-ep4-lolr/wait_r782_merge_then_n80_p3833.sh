#!/usr/bin/env bash
# p3833: wait R782 MERGE_DONE → chall+v4-n80 on R252 GPUs 6,7. Never pkill -f.
set -euo pipefail
log() { echo "[p3833-r782-n80-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
MERGE_DONE=/root/logs/r782_merge.done
CHALL=/root/mining_src/r782-tammy-offline-dpo-hialpha-midrank-lobeta-softctx-megasuperextrasteps-ep4-lolr/lean_chall_n80_r252_gpus67_p3833.sh
LAUNCHED=/root/logs/r782_n80_launched.p3833
mkdir -p /root/logs
[[ -f "$LAUNCHED" ]] && { log "already launched"; exit 0; }
log "armed wait R782 MERGE → n80 GPUs 6,7"
while true; do
  if [[ -f "$MERGE_DONE" ]] && [[ -f /tmp/r782_merged/config.json ]]; then
    n=$(ls /tmp/r782_merged/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
    if [[ "${n:-0}" -ge 16 ]]; then
      log "MERGE_READY shards=$n — launch chall+n80"
      date -u +%Y-%m-%dT%H:%M:%SZ >"$LAUNCHED"
      nohup bash "$CHALL" >/root/logs/p3833_r782_chall_n80_wvk7.log 2>&1 &
      echo $! >/root/logs/p3833_r782_chall_n80.pid
      log "chall pid=$(cat /root/logs/p3833_r782_chall_n80.pid)"
      exit 0
    fi
  fi
  log "waiting merge_done=$([[ -f $MERGE_DONE ]] && echo 1 || echo 0)"
  sleep 30
done
