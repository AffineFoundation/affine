#!/usr/bin/env bash
# p3869: wait R805 MERGE_DONE → chall+v4-n80 on R252 GPUs 4,5. Never pkill -f.
set -euo pipefail
log() { echo "[p3869-r805-n80-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
MERGE_DONE=/root/logs/r805_merge.done
CHALL=/root/mining_src/r805-r252-offline-dpo-hialpha-hirank-hibeta-softctx-megasuperextrasteps-ep4-ultralolr/lean_chall_n80_r252_gpus45_p3869.sh
LAUNCHED=/root/logs/r805_n80_launched.p3869
mkdir -p /root/logs
[[ -f "$LAUNCHED" ]] && { log "already launched"; exit 0; }
log "armed wait R805 MERGE → n80 GPUs 4,5"
while true; do
  if [[ -f "$MERGE_DONE" ]] && [[ -f /tmp/r805_merged/config.json ]]; then
    n=$(ls /tmp/r805_merged/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
    if [[ "${n:-0}" -ge 16 ]]; then
      log "MERGE_READY shards=$n — launch chall+n80"
      date -u +%Y-%m-%dT%H:%M:%SZ >"$LAUNCHED"
      nohup bash "$CHALL" >/root/logs/p3869_r805_chall_n80_wvk7.log 2>&1 &
      echo $! >/root/logs/p3869_r805_chall_n80.pid
      log "chall pid=$(cat /root/logs/p3869_r805_chall_n80.pid)"
      exit 0
    fi
  fi
  log "waiting merge_done=$([[ -f $MERGE_DONE ]] && echo 1 || echo 0)"
  sleep 30
done
