#!/usr/bin/env bash
set -euo pipefail
log() { echo "[p4022-r921-n80-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
MERGE_DONE=/root/logs/r921_merge.done
CHALL=/root/mining_src/r921-tammy-offline-dpo-hialpha-midrank-lobeta-softctx-megasuperextrasteps-ep4-ultralolr/lean_chall_n80_brave_gpu4_p4022.sh
LAUNCHED=/root/logs/r921_n80_launched.p4022
mkdir -p /root/logs
[[ -f "$LAUNCHED" ]] && { log already_launched; exit 0; }
log "armed wait R921 MERGE → n80 :8004"
while true; do
  if [[ -f "$MERGE_DONE" ]] && [[ -f /tmp/r921_merged/config.json ]]; then
    n=$(ls /tmp/r921_merged/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
    if [[ "${n:-0}" -ge 16 ]]; then
      kid=$(curl -sf -m 3 http://127.0.0.1:8001/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || true)
      if ! echo "${kid:-}" | grep -qiE 'vera6|5g4yy75zuz|t6'; then log "king?${kid:-}"; sleep 30; continue; fi
      if [[ ! -x "$CHALL" ]]; then log FATAL_CHALL; sleep 60; continue; fi
      log "MERGE_READY shards=$n — launch chall"
      nohup bash "$CHALL" >/root/logs/p4022_r921_chall_n80.outer.nohup 2>&1 &
      echo $! >/root/logs/p4022_r921_chall_n80.outer.pid
      date -u +%Y-%m-%dT%H:%M:%SZ >"$LAUNCHED"; exit 0
    fi
  fi
  log "waiting merge_done=$([[ -f $MERGE_DONE ]] && echo 1 || echo 0)"; sleep 30
done
