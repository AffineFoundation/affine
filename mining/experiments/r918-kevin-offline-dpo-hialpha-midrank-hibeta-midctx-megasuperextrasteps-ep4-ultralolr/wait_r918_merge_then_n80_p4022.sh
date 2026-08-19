#!/usr/bin/env bash
set -euo pipefail
log() { echo "[p4022-r918-n80-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
MERGE_DONE=/root/logs/r918_merge.done
CHALL=/root/mining_src/r918-kevin-offline-dpo-hialpha-midrank-hibeta-midctx-megasuperextrasteps-ep4-ultralolr/lean_chall_n80_golden_gpus67_p4022.sh
LAUNCHED=/root/logs/r918_n80_launched.p4022
mkdir -p /root/logs
[[ -f "$LAUNCHED" ]] && { log "already launched"; exit 0; }
log "armed wait R918 MERGE → n80 GPUs 6,7 :8003"
while true; do
  if [[ -f "$MERGE_DONE" ]] && [[ -f /tmp/r918_merged/config.json ]]; then
    n=$(ls /tmp/r918_merged/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
    if [[ "${n:-0}" -ge 16 ]]; then
      kid=$(curl -sf -m 3 http://127.0.0.1:8001/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || true)
      if ! echo "${kid:-}" | grep -qiE 'vera6|5g4yy75zuz|t6'; then
        log "MERGE_READY but king id=${kid:-?} not reign36 — wait"; sleep 30; continue
      fi
      if [[ ! -x "$CHALL" ]]; then log "FATAL missing CHALL"; sleep 60; continue; fi
      log "MERGE_READY shards=$n king=$kid — launch chall+n80"
      nohup bash "$CHALL" >/root/logs/p4022_r918_chall_n80.outer.nohup 2>&1 &
      echo $! >/root/logs/p4022_r918_chall_n80.outer.pid
      date -u +%Y-%m-%dT%H:%M:%SZ >"$LAUNCHED"
      log "chall outer pid=$(cat /root/logs/p4022_r918_chall_n80.outer.pid)"; exit 0
    fi
  fi
  log "waiting merge_done=$([[ -f $MERGE_DONE ]] && echo 1 || echo 0)"; sleep 30
done
