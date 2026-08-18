#!/usr/bin/env bash
# Wait for R795 SCP_READY stamp then lean chall+n80 on lunar 6,7.
set -euo pipefail
STAMP=/root/logs/r795_scp_ready.done
LOG=/root/logs/wait_r795_scp_then_chall_p3872.log
LEAN=/root/mining_src/r795-chall/lean_chall_n80_lunar_gpus67_p3872.sh
mkdir -p /root/logs
: >"$LOG"
log(){ echo "[p3872-wait-r795] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }
log "waiting for $STAMP"
while [[ ! -f "$STAMP" ]]; do sleep 15; done
log "stamp=$(cat "$STAMP") launching lean"
exec bash "$LEAN"
