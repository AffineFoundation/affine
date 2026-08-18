#!/usr/bin/env bash
# p3878: wait for R800 SCP_READY then lean chall+n80 on crown GPUs 4,5/:8002.
# Leave R783 on 6,7 alone. Never pkill -f.
set -euo pipefail
STAMP=/root/logs/r800_scp_ready.done
LOG=/root/logs/wait_r800_scp_then_chall_p3878.log
LEAN=/root/mining_src/r800-chall/lean_chall_n80_crown_gpus45_p3878.sh
mkdir -p /root/logs
: >"$LOG"
log(){ echo "[p3878-wait-r800] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }
log "waiting for $STAMP (R783 may run n80 on 6,7 in parallel)"
while [[ ! -f "$STAMP" ]]; do sleep 15; done
log "stamp=$(cat "$STAMP") launching lean"
exec bash "$LEAN"
