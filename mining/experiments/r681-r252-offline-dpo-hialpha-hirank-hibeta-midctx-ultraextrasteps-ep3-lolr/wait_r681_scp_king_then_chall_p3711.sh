#!/usr/bin/env bash
# p3711: wait R681 SCP_READY + reign34 king on golden → lean chall+v4-n80 GPUs 4,5/:8003.
# Never pkill -f. Leave R637 on 6,7/:8004 alone.
set -euo pipefail

log() { echo "[p3711-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a /root/logs/p3711_r681_wait.log; }

SCP_DONE=/root/logs/r681_scp_ready.done
KING_DONE=/root/logs/swap_king_reign34_golden_p3711.done
LAUNCHED=/root/logs/r681_chall_n80_launched.p3711
LEAN=/root/mining_src/r681-chall/lean_chall_n80_golden_gpus45_p3711.sh
mkdir -p /root/logs
: >>/root/logs/p3711_r681_wait.log

log "armed: wait R681 SCP + reign34 king → chall 4,5/:8003 (timeout 4h)"
for i in $(seq 1 1440); do
  if [[ -f "$LAUNCHED" ]]; then
    log "already launched — exit"
    exit 0
  fi

  if [[ ! -f "$KING_DONE" ]]; then
    code=$(curl -s -o /tmp/king_models_p3711.json -w "%{http_code}" http://127.0.0.1:8001/v1/models || echo 000)
    id=$(python3 -c "import json;print(json.load(open('/tmp/king_models_p3711.json')).get('data',[{}])[0].get('id','none'))" 2>/dev/null || echo none)
    if echo "$id" | grep -qiE '5Dku3dYp9j|cryptoDev23|hk8161'; then
      date -u +%Y-%m-%dT%H:%M:%SZ >"$KING_DONE"
      log "king already reign34 id=$id — mark KING_DONE"
    fi
  fi

  if [[ -f "$SCP_DONE" ]] && [[ -f "$KING_DONE" ]]; then
    if [[ -f /tmp/r681_merged/config.json ]] \
      && [[ $(ls /tmp/r681_merged/model-*-of-*.safetensors 2>/dev/null | wc -l) -ge 16 ]]; then
      code=$(curl -s -o /tmp/king_models_p3711b.json -w "%{http_code}" http://127.0.0.1:8001/v1/models || echo 000)
      id=$(python3 -c "import json;print(json.load(open('/tmp/king_models_p3711b.json')).get('data',[{}])[0].get('id','none'))" 2>/dev/null || echo none)
      if [[ "$code" = "200" ]] && echo "$id" | grep -qiE '5Dku3dYp9j|cryptoDev23|hk8161'; then
        date -u +%Y-%m-%dT%H:%M:%SZ >"$LAUNCHED"
        log "GO king=$id — launch lean chall+v4-n80"
        nohup bash "$LEAN" >/root/logs/p3711_r681_lean_outer.nohup 2>&1 &
        echo $! >/root/logs/p3711_r681_lean_outer.pid
        log "lean outer pid=$(cat /root/logs/p3711_r681_lean_outer.pid)"
        exit 0
      else
        log "markers ok but king not ready code=$code id=$id (i=$i)"
      fi
    else
      log "SCP done marker but merge incomplete (i=$i)"
    fi
  else
    s=missing; k=missing
    [[ -f "$SCP_DONE" ]] && s=ok
    [[ -f "$KING_DONE" ]] && k=ok
    n=$(ls /tmp/r681_merged/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
    [[ $((i % 6)) -eq 0 ]] && log "poll i=$i scp=$s king=$k shards=${n:-0}"
  fi
  sleep 10
done
log "TIMEOUT waiting scp+king"
exit 1
