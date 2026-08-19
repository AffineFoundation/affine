#!/usr/bin/env bash
# p4046 host outer: (1) fix hung R337 chall Triton → n80 (2) relay R927 → R337:8003 → n80 after stamp.
set -euo pipefail
ROOT=/home/const/subnet120/mining
LOGDIR=$ROOT/experiments/fleet-rent/logs
mkdir -p "$LOGDIR" "$ROOT/.ralph"
LOG=$LOGDIR/p4046_r337_fix_and_r927_relay.outer.log
: >"$LOG"
log() { echo "[p4046-outer] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=45 -o BatchMode=yes
  -o ServerAliveInterval=15 -o ServerAliveCountMax=40 -o TCPKeepAlive=yes)
R337_HOST=150.136.46.118; R337_PORT=20300

R337_LEAN=$ROOT/experiments/r337-marsplan-online-dpo-hilr/lean_chall_n80_triton_reseed_p4046.sh
R927_LEAN=$ROOT/experiments/r927-cryptodev23-offline-dpo-hialpha-midrank-midlobeta-midctx-megasuperextrasteps-ep4-ultralolr/lean_chall_n80_r337_gpus67_p4046.sh
RELAY=$ROOT/experiments/fleet-rent/host_relay_r927_to_r337_p4046.sh

chmod +x "$R337_LEAN" "$R927_LEAN" "$RELAY"

log "scp lean scripts → R337"
scp "${SSH_OPTS[@]}" -P "$R337_PORT" "$R337_LEAN" "root@$R337_HOST:/root/lean_chall_n80_triton_reseed_p4046.sh"
scp "${SSH_OPTS[@]}" -P "$R337_PORT" "$R927_LEAN" "root@$R337_HOST:/root/lean_chall_n80_r927_r337_gpus67_p4046.sh"
ssh "${SSH_OPTS[@]}" -p "$R337_PORT" "root@$R337_HOST" \
  'chmod +x /root/lean_chall_n80_triton_reseed_p4046.sh /root/lean_chall_n80_r927_r337_gpus67_p4046.sh && mkdir -p /root/logs /root/mining_src/r337-marsplan-online-dpo-hilr && cp -f /root/lean_chall_n80_triton_reseed_p4046.sh /root/mining_src/r337-marsplan-online-dpo-hilr/'

log "launch R337 Triton-reseed chall→n80"
ssh "${SSH_OPTS[@]}" -p "$R337_PORT" "root@$R337_HOST" \
  'nohup bash /root/lean_chall_n80_triton_reseed_p4046.sh >/root/logs/p4046_r337_outer.nohup 2>&1 & echo $! >/root/logs/p4046_r337_outer.pid; echo launched:$(cat /root/logs/p4046_r337_outer.pid)'

log "start R927 host-relay background"
nohup bash "$RELAY" >>"$LOGDIR/host_relay_r927_to_r337_p4046.outer.nohup" 2>&1 &
echo $! >"$LOGDIR/host_relay_r927_to_r337_p4046.outer.pid"
log "relay outer pid=$(cat $LOGDIR/host_relay_r927_to_r337_p4046.outer.pid)"

log "arm R927 wait→lean after stamp"
nohup bash -c '
set -euo pipefail
LOGDIR=/home/const/subnet120/mining/experiments/fleet-rent/logs
DONE=$LOGDIR/host_relay_r927_to_r337_p4046.done
SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=45 -o BatchMode=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=40 -o TCPKeepAlive=yes)
R337_HOST=150.136.46.118; R337_PORT=20300
echo "[p4046-r927-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) waiting for $DONE" >>$LOGDIR/p4046_r927_wait_lean.log
for i in $(seq 1 720); do
  [[ -f "$DONE" ]] && break
  sleep 30
done
[[ -f "$DONE" ]] || { echo "[p4046-r927-wait] FATAL no stamp" >>$LOGDIR/p4046_r927_wait_lean.log; exit 1; }
echo "[p4046-r927-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) stamp ok — launch lean" >>$LOGDIR/p4046_r927_wait_lean.log
ssh "${SSH_OPTS[@]}" -p "$R337_PORT" "root@$R337_HOST" \
  "test -f /root/logs/r927_scp_ready.done && test -f /tmp/r927_merged/config.json && ls /tmp/r927_merged/model-*-of-*.safetensors | wc -l | grep -qE \"^(1[6-9]|[2-9][0-9])$\" && nohup bash /root/lean_chall_n80_r927_r337_gpus67_p4046.sh >/root/logs/p4046_r927_outer.nohup 2>&1 & echo \$! >/root/logs/p4046_r927_outer.pid; echo launched:\$(cat /root/logs/p4046_r927_outer.pid)" \
  >>$LOGDIR/p4046_r927_wait_lean.log 2>&1
' >>"$LOGDIR/p4046_r927_wait_lean.log" 2>&1 &
echo $! >"$LOGDIR/p4046_r927_wait_lean.pid"
log "r927 wait-lean pid=$(cat $LOGDIR/p4046_r927_wait_lean.pid)"
log "DONE arm"
