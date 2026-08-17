#!/usr/bin/env bash
# p3688: wait R655 SCP_READY + reign34 king → chall 4,5/:8003 (v4 n80).
# Never pkill -f. Leave R537 :8002 alone.
set -euo pipefail
log() { echo "[p3688-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a /root/logs/p3688_r655_wait.log; }
KING_DONE=/root/logs/swap_king_reign34_p3585.done
SCP_DONE=/root/logs/r655_scp_ready.done
LAUNCHED=/root/logs/r655_chall_n80_launched.p3688
LEAN=/root/mining_src/r655-chall/lean_chall_n80_lunar_gpus45_p3688.sh
mkdir -p /root/logs
: >>/root/logs/p3688_r655_wait.log
log "armed: wait king READY + r655 SCP_READY → chall 4,5/:8003 (timeout 6h)"
for i in $(seq 1 2160); do
  if [[ -f "$LAUNCHED" ]]; then
    log "already launched — exit"
    exit 0
  fi
  king_ok=0
  if [[ -f "$KING_DONE" ]]; then
    king_ok=1
  else
    code=$(curl -s -o /tmp/king_models_p3688.json -w "%{http_code}" http://127.0.0.1:8001/v1/models || echo 000)
    id=$(python3 -c "import json;print(json.load(open('/tmp/king_models_p3688.json')).get('data',[{}])[0].get('id','none'))" 2>/dev/null || echo none)
    if [[ "$code" = "200" ]] && echo "$id" | grep -qiE '5Dku3dYp9j|cryptoDev23|hk8161'; then
      king_ok=1
    fi
  fi
  if [[ "$king_ok" -eq 1 ]] && [[ -f "$SCP_DONE" ]]; then
    if [[ -f /tmp/r655_merged/config.json ]] \
      && [[ $(ls /tmp/r655_merged/model-*-of-*.safetensors 2>/dev/null | wc -l) -ge 16 ]]; then
      code=$(curl -s -o /tmp/king_models_p3688.json -w "%{http_code}" http://127.0.0.1:8001/v1/models || echo 000)
      id=$(python3 -c "import json;print(json.load(open('/tmp/king_models_p3688.json')).get('data',[{}])[0].get('id','none'))" 2>/dev/null || echo none)
      if [[ "$code" = "200" ]] && echo "$id" | grep -qiE '5Dku3dYp9j|cryptoDev23|hk8161'; then
        date -u +%Y-%m-%dT%H:%M:%SZ >"$LAUNCHED"
        log "GO king=$id — launch lean chall+v4-n80"
        nohup bash "$LEAN" >/root/logs/p3688_r655_lean_outer.nohup 2>&1 &
        echo $! >/root/logs/p3688_r655_lean_outer.pid
        log "lean outer pid=$(cat /root/logs/p3688_r655_lean_outer.pid)"
        exit 0
      else
        log "markers ok but king not ready code=$code id=$id (i=$i)"
      fi
    else
      log "SCP done marker but merge incomplete (i=$i)"
    fi
  else
    k=missing; s=missing
    [[ "$king_ok" -eq 1 ]] && k=ok
    [[ -f "$SCP_DONE" ]] && s=ok
    n=$(ls /tmp/r655_merged/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
    [[ $((i % 6)) -eq 0 ]] && log "poll i=$i king_done=$k scp_done=$s shards=${n:-0}"
  fi
  sleep 10
done
log "TIMEOUT waiting king+scp"
exit 1
