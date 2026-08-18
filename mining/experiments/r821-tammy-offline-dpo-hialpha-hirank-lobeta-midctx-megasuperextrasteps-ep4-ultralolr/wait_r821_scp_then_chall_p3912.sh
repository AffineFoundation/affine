#!/usr/bin/env bash
# p3912: wait r821 SCP_READY on crown, claim first free chall slot after R827/R829 harvest.
# Never pkill -f. Never steal a live chall (port quiet + GPUs nearly empty).
set -euo pipefail
STAMP=/root/logs/r821_scp_ready.done
LOG=/root/logs/wait_r821_scp_then_chall_p3912.log
LEAN=/root/mining_src/r821-chall/lean_chall_n80_crown_slot_p3904.sh
CLAIM=/root/logs/r821_chall_slot_claimed.p3912
mkdir -p /root/logs
: >"$LOG"
log(){ echo "[p3912-wait-r821] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }
[[ -f "$CLAIM" ]] && { log "already claimed $(cat "$CLAIM")"; exit 0; }
log "waiting for $STAMP"
while [[ ! -f "$STAMP" ]]; do sleep 20; done
log "stamp=$(cat "$STAMP") — polling for free chall slot"

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
    exec env R821_GPUS=4,5 R821_CHALL_PORT=8002 bash "$LEAN"
  fi
  if port_quiet 8003 && gpus_free 6,7; then
    echo "6,7:8003" >"$CLAIM"
    log "CLAIM GPUs=6,7 port=8003"
    exec env R821_GPUS=6,7 R821_CHALL_PORT=8003 bash "$LEAN"
  fi
  r827=$( [[ -f /root/affine_data/r827_sim_progress_reign35_wvk7.json ]] && python3 -c "import json;d=json.load(open('/root/affine_data/r827_sim_progress_reign35_wvk7.json'));print(d.get('challenger','?'))" 2>/dev/null || echo '?' )
  r829=$( [[ -f /root/affine_data/r829_sim_progress_reign35_wvk7.json ]] && python3 -c "import json;d=json.load(open('/root/affine_data/r829_sim_progress_reign35_wvk7.json'));print(d.get('challenger','?'))" 2>/dev/null || echo '?' )
  log "waiting free slot (r827~$r827/80 r829~$r829/80)"
  sleep 30
done
