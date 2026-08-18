#!/usr/bin/env bash
# p3917: wait R835 SCP_READY on lunar, claim first free chall slot (4,5/:8002 or 6,7/:8003).
# Never steal live R818/R826 n80. Never pkill -f.
set -euo pipefail
STAMP=/root/logs/r835_scp_ready.done
LOG=/root/logs/wait_r835_scp_then_chall_lunar_p3917.log
LEAN=/root/mining_src/r835-chall/lean_chall_n80_lunar_slot_r835_p3917.sh
CLAIM=/root/logs/r835_chall_slot_claimed.p3917
mkdir -p /root/logs
: >"$LOG"
log(){ echo "[p3917-r835-slot] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }
[[ -f "$CLAIM" ]] && { log "already claimed $(cat "$CLAIM")"; exit 0; }
log "waiting for $STAMP"
while [[ ! -f "$STAMP" ]]; do sleep 20; done
log "stamp=$(cat "$STAMP") — polling free chall slot"

port_quiet() {
  local port=$1
  ! ss -lptn "sport = :$port" 2>/dev/null | grep -q LISTEN
}
gpus_free() {
  local ids=$1
  local used
  used=$(nvidia-smi -i "$ids" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END{print s+0}')
  [[ "${used:-999999}" -lt 4000 ]]
}

while true; do
  if port_quiet 8002 && gpus_free 4,5; then
    echo "4,5:8002" >"$CLAIM"
    log "CLAIM GPUs=4,5 port=8002"
    exec env R835_GPUS=4,5 R835_CHALL_PORT=8002 bash "$LEAN"
  fi
  if port_quiet 8003 && gpus_free 6,7; then
    echo "6,7:8003" >"$CLAIM"
    log "CLAIM GPUs=6,7 port=8003"
    exec env R835_GPUS=6,7 R835_CHALL_PORT=8003 bash "$LEAN"
  fi
  log "waiting free slot (R818/R826 trains or n80)"
  sleep 30
done
