#!/usr/bin/env bash
# p3880: wait for R801 SCP_READY then lean chall+n80 on crown GPUs 6,7/:8003.
# Leave R800 on 4,5 alone. Never pkill -f.
set -euo pipefail
STAMP=/root/logs/r801_scp_ready.done
LOG=/root/logs/wait_r801_scp_then_chall_p3880.log
LEAN=/root/mining_src/r801-chall/lean_chall_n80_crown_gpus67_p3880.sh
mkdir -p /root/logs
: >"$LOG"
log(){ echo "[p3880-wait-r801] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }
log "waiting for $STAMP (R800 may run n80 on 4,5 in parallel)"
while [[ ! -f "$STAMP" ]]; do sleep 15; done
log "stamp=$(cat "$STAMP") launching lean"
exec bash "$LEAN"
