#!/usr/bin/env bash
# p3963: parent cold-TK died when TP2 king was reaped; wait TP1 king READY then arm R848/R849 n80.
# Never pkill -f. Teacher must already be LIVE on :8000.
set -euo pipefail
LOG=/root/logs/p3963_arm_r848_r849.log
mkdir -p /root/logs /root/affine_data
: >"$LOG"
log() { echo "[p3963-arm] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

KING_PIDF=/root/logs/vllm_king.pid
KING_LOG=/root/logs/vllm_king.log

curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null || { log "FATAL teacher :8000 down"; exit 1; }
log "teacher READY"

ready=0
for i in $(seq 1 480); do
  if curl -sf -m 3 http://127.0.0.1:8001/v1/models >/dev/null 2>&1; then
    ready=1; log "king_READY poll=$i"; break
  fi
  if [[ -f "$KING_PIDF" ]]; then
    pid=$(cat "$KING_PIDF")
    if ! kill -0 "$pid" 2>/dev/null; then
      log "ERROR king died"; tail -80 "$KING_LOG" | tee -a "$LOG"; exit 1
    fi
  fi
  (( i % 12 == 0 )) && log "wait king :8001 iter=$i last=$(tail -1 "$KING_LOG" 2>/dev/null | cut -c1-100)"
  sleep 5
done
[[ "$ready" -eq 1 ]] || { log "ERROR king not ready"; exit 2; }

kid=$(curl -sf -m 5 http://127.0.0.1:8001/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
log "king id=$kid"
echo "$kid" | grep -qiE 'vera6|5g4yy75zuz|t6' || { log "ERROR king not vera"; exit 5; }
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/brave_tk_ready_p3956.done
log "TK READY — launch R848+R849 lean challs"

for s in /root/mining_src/fleet-rent/lean_chall_n80_brave_r848_gpus45_p3956.sh \
         /root/mining_src/fleet-rent/lean_chall_n80_brave_r849_gpus67_p3956.sh; do
  [[ -x "$s" ]] || chmod +x "$s"
  tag=$(basename "$s" .sh)
  # skip if already armed and alive
  if [[ -f /root/logs/${tag}.outer.pid ]]; then
    opid=$(cat /root/logs/${tag}.outer.pid)
    if kill -0 "$opid" 2>/dev/null; then
      log "keep live $tag pid=$opid"; continue
    fi
  fi
  nohup bash "$s" >/root/logs/${tag}.outer.log 2>&1 &
  echo $! >/root/logs/${tag}.outer.pid
  log "armed $s pid=$(cat /root/logs/${tag}.outer.pid)"
done

log "DONE arm — watch /root/affine_data/r848_sim_* /root/affine_data/r849_sim_*"
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p3963_brave_r848_r849_armed.done
