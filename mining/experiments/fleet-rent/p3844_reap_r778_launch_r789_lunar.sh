#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/const/subnet120/mining
cd "$ROOT"
source /home/const/subnet120/.venv/bin/activate
LOG=$ROOT/experiments/fleet-rent/logs/p3844_reap_r778_launch_r789_lunar.log
mkdir -p "$(dirname "$LOG")"
: >"$LOG"
log(){ echo "[p3844-r789] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }
HUID=lunar-wolf-be
EXP=r789-marsplan-offline-dpo-hialpha-midrank-lobeta-softctx-megasuperextrasteps-ep4-ultralolr
log "scp experiment tree"
# tar upload to /tmp then extract (lium scp targets source dest)
tar czf /tmp/r789_p3844.tgz -C experiments "$EXP"
lium scp "$HUID" /tmp/r789_p3844.tgz /tmp/r789_p3844.tgz >>"$LOG" 2>&1
lium scp "$HUID" experiments/fleet-rent/p3844_onpod_reap_r778_launch_r789.sh /tmp/p3844_onpod_reap_r778_launch_r789.sh >>"$LOG" 2>&1
log "exec onpod"
timeout 300 lium exec "$HUID" 'set -e
mkdir -p /root/mining_src
tar xzf /tmp/r789_p3844.tgz -C /root/mining_src
cp -f /tmp/p3844_onpod_reap_r778_launch_r789.sh /root/p3844_onpod_reap_r778_launch_r789.sh
chmod +x /root/p3844_onpod_reap_r778_launch_r789.sh
bash /root/p3844_onpod_reap_r778_launch_r789.sh' 2>&1 | tee -a "$LOG"
log DONE
