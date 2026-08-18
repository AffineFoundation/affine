#!/usr/bin/env bash
# p3844 host driver: scp patched leans + on-pod fix, then exec.
set -euo pipefail
ROOT=/home/const/subnet120/mining
cd "$ROOT"
source /home/const/subnet120/.venv/bin/activate
LOGDIR=$ROOT/experiments/fleet-rent/logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/p3844_fix_golden_r773_r774_port_clash.log
: >"$LOG"
log() { echo "[p3844-golden] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

HUID=golden-comet-78
R773_LEAN=$ROOT/experiments/r773-r252-offline-dpo-hialpha-midrank-midbeta-midctx-megasuperextrasteps-ep4-lolr/lean_chall_n80_golden_gpus67_p3822.sh
R774_LEAN=$ROOT/experiments/r774-r252-offline-dpo-hialpha-midrank-lobeta-midctx-megasuperextrasteps-ep4-lolr/lean_chall_n80_golden_gpus45_p3822.sh
ONPOD=$ROOT/experiments/fleet-rent/p3844_onpod_fix_golden_r773_r774.sh
R773_DST=/root/mining_src/r773-r252-offline-dpo-hialpha-midrank-midbeta-midctx-megasuperextrasteps-ep4-lolr/lean_chall_n80_golden_gpus67_p3822.sh
R774_DST=/root/mining_src/r774-r252-offline-dpo-hialpha-midrank-lobeta-midctx-megasuperextrasteps-ep4-lolr/lean_chall_n80_golden_gpus45_p3822.sh

log "START scp + exec port-clash fix"
lium scp "$R774_LEAN" "${HUID}:${R774_DST}" >>"$LOG" 2>&1
lium scp "$R773_LEAN" "${HUID}:${R773_DST}" >>"$LOG" 2>&1
lium scp "$ONPOD" "${HUID}:/root/p3844_onpod_fix_golden_r773_r774.sh" >>"$LOG" 2>&1
log "scp done; exec onpod (may take ~15m for both CHALL_READY)"
# block_until long enough for vLLM load
timeout 1200 lium exec "$HUID" 'bash /root/p3844_onpod_fix_golden_r773_r774.sh' 2>&1 | tee -a "$LOG"
log "DONE"
