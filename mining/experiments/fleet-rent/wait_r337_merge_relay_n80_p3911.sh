#!/usr/bin/env bash
# p3911: wait R337 merge.done on gentle-shark → host-relay → lunar → arm slot-claimer n80.
# Never pkill -f.
set -euo pipefail
ROOT=/home/const/subnet120/mining
LOGDIR=$ROOT/experiments/fleet-rent/logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/wait_r337_merge_relay_n80_p3911.log
: >"$LOG"
log() { echo "[p3911-r337-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -o BatchMode=yes)
SRC_HOST=86.38.182.67; SRC_PORT=20295
LUNAR_HOST=150.136.46.118; LUNAR_PORT=20299
RELAY=$ROOT/experiments/fleet-rent/host_relay_r337_to_lunar_p3911.sh
DONE=$LOGDIR/wait_r337_merge_relay_n80_p3911.done
[[ -f "$DONE" ]] && { log "already done"; exit 0; }

log "WAIT for R337 /root/logs/r337_merge.done"
while true; do
  st=$(ssh "${SSH_OPTS[@]}" -p "$SRC_PORT" "root@$SRC_HOST" \
    '[[ -f /root/logs/r337_merge.done ]] && echo YES || echo NO' 2>/dev/null || echo ERR)
  [[ "$st" == "YES" ]] && break
  step=$(ssh "${SSH_OPTS[@]}" -p "$SRC_PORT" "root@$SRC_HOST" \
    'grep -o "\"step\": [0-9]*" /root/logs/r337_train.nohup 2>/dev/null | tail -1' 2>/dev/null || true)
  log "waiting merge.done (train $step)"
  sleep 45
done
log "merge.done seen — start relay"
nohup bash "$RELAY" >"$LOGDIR/host_relay_r337_to_lunar_p3911.outer.nohup" 2>&1 &
echo $! >"$LOGDIR/host_relay_r337_to_lunar_p3911.outer.pid"
log "relay pid=$(cat "$LOGDIR/host_relay_r337_to_lunar_p3911.outer.pid")"
wait "$(cat "$LOGDIR/host_relay_r337_to_lunar_p3911.outer.pid")"
[[ -f "$LOGDIR/host_relay_r337_to_lunar_p3911.done" ]] || { log "FATAL relay failed"; exit 1; }

log "arm lunar slot-claimer for R337 n80"
scp "${SSH_OPTS[@]}" -P "$LUNAR_PORT" \
  "$ROOT/experiments/fleet-rent/lean_chall_n80_lunar_slot_r337_p3911.sh" \
  "$ROOT/experiments/fleet-rent/wait_r337_scp_then_chall_lunar_p3911.sh" \
  "root@$LUNAR_HOST:/root/mining_src/r337-chall/"
ssh "${SSH_OPTS[@]}" -p "$LUNAR_PORT" "root@$LUNAR_HOST" bash -s <<'REMOTE'
set -euo pipefail
mkdir -p /root/mining_src/r337-chall /root/logs
chmod +x /root/mining_src/r337-chall/*.sh
[[ -f /root/logs/r337_chall_slot_claimed.p3911 ]] && exit 0
nohup bash /root/mining_src/r337-chall/wait_r337_scp_then_chall_lunar_p3911.sh \
  >/root/logs/wait_r337_scp_then_chall_lunar_p3911.nohup 2>&1 &
echo $! >/root/logs/wait_r337_scp_then_chall_lunar_p3911.pid
echo "armed pid=$(cat /root/logs/wait_r337_scp_then_chall_lunar_p3911.pid)"
REMOTE
date -u +%Y-%m-%dT%H:%M:%SZ >"$DONE"
log "DONE — R337 merge→relay→lunar slot-waiter armed"
