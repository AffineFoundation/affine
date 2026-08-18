#!/usr/bin/env bash
# Wait for R794 SCP_READY stamp then lean chall+n80 on lunar 4,5.
set -euo pipefail
STAMP=/root/logs/r794_scp_ready.done
LOG=/root/logs/wait_r794_scp_then_chall_p3872.log
LEAN=/root/mining_src/r794-chall/lean_chall_n80_lunar_gpus45_p3872.sh
mkdir -p /root/logs
: >"$LOG"
log(){ echo "[p3872-wait-r794] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }
log "waiting for $STAMP"
while [[ ! -f "$STAMP" ]]; do sleep 15; done
log "stamp=$(cat "$STAMP") launching lean"
exec bash "$LEAN"
