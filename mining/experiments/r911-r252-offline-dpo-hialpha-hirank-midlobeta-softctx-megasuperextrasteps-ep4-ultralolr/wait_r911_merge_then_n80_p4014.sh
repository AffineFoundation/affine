#!/usr/bin/env bash
# p4014: stamp LAUNCHED only after outer stays alive; require -x CHALL (bad path → no stamp).
set -euo pipefail
log() { echo "[p4014-r911-n80-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
MERGE_DONE=/root/logs/r911_merge.done
CHALL=/root/mining_src/r911-r252-offline-dpo-hialpha-hirank-midlobeta-softctx-megasuperextrasteps-ep4-ultralolr/lean_chall_n80_r252_gpus67_p4014.sh
LAUNCHED=/root/logs/r911_n80_launched.p4014
mkdir -p /root/logs
if [[ -f "$LAUNCHED" ]]; then
  if curl -sf -m 2 http://127.0.0.1:8003/v1/models >/dev/null 2>&1 \
     || [[ -f /root/logs/p4014_r911_chall_n80_wvk7.log ]]; then
    log "already launched (live or log present)"; exit 0
  fi
  log "stale LAUNCHED without chall — clear and relaunch"
  rm -f "$LAUNCHED"
fi
log "armed wait R911 MERGE → n80 GPUs 6,7 :8003"
while true; do
  if [[ -f "$MERGE_DONE" ]] && [[ -f /tmp/r911_merged/config.json ]]; then
    n=$(ls /tmp/r911_merged/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
    if [[ "${n:-0}" -ge 16 ]]; then
      kid=$(curl -sf -m 3 http://127.0.0.1:8001/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || true)
      if ! echo "${kid:-}" | grep -qiE 'vera6|5g4yy75zuz|t6'; then
        log "MERGE_READY but king id=${kid:-?} not reign36 — wait"
        sleep 30
        continue
      fi
      if [[ ! -x "$CHALL" ]]; then
        log "FATAL missing executable CHALL=$CHALL"
        sleep 60
        continue
      fi
      log "MERGE_READY shards=$n king=$kid — launch $CHALL"
      nohup bash "$CHALL" >/root/logs/p4014_r911_chall_n80.outer.nohup 2>&1 &
      echo $! >/root/logs/p4014_r911_chall_n80.outer.pid
      sleep 2
      if ! kill -0 "$(cat /root/logs/p4014_r911_chall_n80.outer.pid)" 2>/dev/null; then
        log "FATAL outer died:"; cat /root/logs/p4014_r911_chall_n80.outer.nohup
        exit 1
      fi
      date -u +%Y-%m-%dT%H:%M:%SZ >"$LAUNCHED"
      log "outer pid=$(cat /root/logs/p4014_r911_chall_n80.outer.pid) LAUNCHED after live"
      exit 0
    fi
  fi
  log "waiting merge_done=$([[ -f $MERGE_DONE ]] && echo 1 || echo 0)"
  sleep 30
done
