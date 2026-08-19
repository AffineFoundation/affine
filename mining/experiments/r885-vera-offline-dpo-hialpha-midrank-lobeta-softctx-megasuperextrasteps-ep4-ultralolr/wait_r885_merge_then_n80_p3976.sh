#!/usr/bin/env bash
set -euo pipefail
log() { echo "[p3976-r885-n80-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
MERGE_DONE=/root/logs/r885_merge.done
CHALL=/root/mining_src/r885-vera-offline-dpo-hialpha-midrank-lobeta-softctx-megasuperextrasteps-ep4-ultralolr/lean_chall_n80_crown_gpus45_p3935.sh
LAUNCHED=/root/logs/r885_n80_launched.p4002
mkdir -p /root/logs
[[ -f "$LAUNCHED" ]] && { log "already launched"; exit 0; }
log "armed wait R885 MERGE → n80 GPUs 4,5 (require vera king)"
while true; do
  if [[ -f "$MERGE_DONE" ]] && [[ -f /tmp/r885_merged/config.json ]]; then
    n=$(ls /tmp/r885_merged/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
    kid=$(curl -sf -m 3 http://127.0.0.1:8001/v1/models 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || true)
    if [[ "${n:-0}" -ge 16 ]] && [[ "$kid" == *vera6* ]]; then
      log "MERGE_READY shards=$n king=$kid — launch chall+n80"
      date -u +%Y-%m-%dT%H:%M:%SZ >"$LAUNCHED"
      nohup bash "$CHALL" >/root/logs/p4002_r885_chall_n80_wvk7.log 2>&1 &
      echo $! >/root/logs/p4002_r885_chall_n80.pid
      log "chall pid=$(cat /root/logs/p4002_r885_chall_n80.pid)"
      exit 0
    fi
    log "waiting shards=${n:-0} king=${kid:-none}"
  else
    log "waiting merge_done=$([[ -f $MERGE_DONE ]] && echo 1 || echo 0)"
  fi
  sleep 30
done
