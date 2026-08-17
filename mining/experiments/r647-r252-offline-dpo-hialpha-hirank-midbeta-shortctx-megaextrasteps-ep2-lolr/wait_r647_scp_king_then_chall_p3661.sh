#!/usr/bin/env bash
# p3661: wait R647 SCP_READY + reign34 king → chall 4,5/:8003.
# Never pkill -f. Leave R637 :8004 alone.
set -euo pipefail
log() { echo "[p3661-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a /root/logs/p3661_r647_wait.log; }
KING_DONE=/root/logs/swap_king_reign34_p3585.done
SCP_DONE=/root/logs/r647_scp_ready.done
LAUNCHED=/root/logs/r647_chall_n80_launched.p3661
LEAN=/root/mining_src/r647-chall/lean_chall_n80_golden_gpus45_p3661.sh
mkdir -p /root/logs
: >>/root/logs/p3661_r647_wait.log
log "armed: wait king READY + r647 SCP_READY → chall 4,5/:8003 (timeout 4h)"
for i in $(seq 1 1440); do
  if [[ -f "$LAUNCHED" ]]; then
    log "already launched — exit"
    exit 0
  fi
  if [[ -f "$KING_DONE" ]] && [[ -f "$SCP_DONE" ]]; then
    if [[ -f /tmp/r647_merged/config.json ]] \
      && [[ $(ls /tmp/r647_merged/model-*-of-*.safetensors 2>/dev/null | wc -l) -ge 16 ]]; then
      code=$(curl -s -o /tmp/king_models_p3661.json -w "%{http_code}" http://127.0.0.1:8001/v1/models || echo 000)
      id=$(python3 -c "import json;print(json.load(open('/tmp/king_models_p3661.json')).get('data',[{}])[0].get('id','none'))" 2>/dev/null || echo none)
      if [[ "$code" = "200" ]] && echo "$id" | grep -qiE '5Dku3dYp9j|cryptoDev23|hk8161'; then
        date -u +%Y-%m-%dT%H:%M:%SZ >"$LAUNCHED"
        log "GO king=$id — launch lean chall+n80"
        nohup bash "$LEAN" >/root/logs/p3661_r647_lean_outer.nohup 2>&1 &
        echo $! >/root/logs/p3661_r647_lean_outer.pid
        log "lean outer pid=$(cat /root/logs/p3661_r647_lean_outer.pid)"
        exit 0
      else
        log "markers ok but king not ready code=$code id=$id (i=$i)"
      fi
    else
      log "SCP done marker but merge incomplete (i=$i)"
    fi
  else
    k=missing; s=missing
    [[ -f "$KING_DONE" ]] && k=ok
    [[ -f "$SCP_DONE" ]] && s=ok
    n=$(ls /tmp/r647_merged/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
    [[ $((i % 6)) -eq 0 ]] && log "poll i=$i king_done=$k scp_done=$s shards=${n:-0}"
  fi
  sleep 10
done
log "TIMEOUT waiting king+scp"
exit 1
