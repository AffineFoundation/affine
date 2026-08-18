#!/usr/bin/env bash
# p3917: wait R835 merge.done on gentle-shark → host-relay → lunar SIZE_OK stamp.
# Slot-claimer armed after R818 frees :8003 (or free :8002). Never pkill -f.
set -euo pipefail
ROOT=/home/const/subnet120/mining
LOGDIR=$ROOT/experiments/fleet-rent/logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/wait_r835_merge_relay_n80_p3917.log
: >"$LOG"
log() { echo "[p3917-r835-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -o BatchMode=yes)
SRC_HOST=86.38.182.67; SRC_PORT=20295
LUNAR_HOST=150.136.46.118; LUNAR_PORT=20299
RELAY=$ROOT/experiments/fleet-rent/host_relay_r835_to_lunar_p3917.sh
DONE=$LOGDIR/wait_r835_merge_relay_n80_p3917.done
[[ -f "$DONE" ]] && { log "already done"; exit 0; }

log "WAIT for R835 /root/logs/r835_merge.done"
while true; do
  st=$(ssh "${SSH_OPTS[@]}" -p "$SRC_PORT" "root@$SRC_HOST" \
    '[[ -f /root/logs/r835_merge.done ]] && echo YES || echo NO' 2>/dev/null || echo ERR)
  [[ "$st" == "YES" ]] && break
  step=$(ssh "${SSH_OPTS[@]}" -p "$SRC_PORT" "root@$SRC_HOST" \
    'grep -o "\"step\": [0-9]*" /root/logs/r835_train.nohup 2>/dev/null | tail -1' 2>/dev/null || true)
  log "waiting merge.done (train $step)"
  sleep 45
done
log "merge.done seen — start relay"
nohup bash "$RELAY" >"$LOGDIR/host_relay_r835_to_lunar_p3917.outer.nohup" 2>&1 &
echo $! >"$LOGDIR/host_relay_r835_to_lunar_p3917.outer.pid"
log "relay pid=$(cat "$LOGDIR/host_relay_r835_to_lunar_p3917.outer.pid")"
wait "$(cat "$LOGDIR/host_relay_r835_to_lunar_p3917.outer.pid")"
[[ -f "$LOGDIR/host_relay_r835_to_lunar_p3917.done" ]] || { log "FATAL relay failed"; exit 1; }

# Arm lunar slot-claimer (wait scp_ready → claim free :8002/:8003)
ssh "${SSH_OPTS[@]}" -p "$LUNAR_PORT" "root@$LUNAR_HOST" "mkdir -p /root/mining_src/r835-chall /root/logs"
scp "${SSH_OPTS[@]}" -P "$LUNAR_PORT" \
  "$ROOT/experiments/fleet-rent/wait_r835_scp_then_chall_lunar_p3917.sh" \
  "$ROOT/experiments/fleet-rent/lean_chall_n80_lunar_slot_r835_p3917.sh" \
  "root@$LUNAR_HOST:/root/mining_src/r835-chall/"
ssh "${SSH_OPTS[@]}" -p "$LUNAR_PORT" "root@$LUNAR_HOST" bash -s <<'REMOTE'
set -euo pipefail
mkdir -p /root/mining_src/r835-chall /root/logs
chmod +x /root/mining_src/r835-chall/*.sh
if [[ -f /root/logs/wait_r835_scp_then_chall_lunar_p3917.pid ]]; then
  opid=$(cat /root/logs/wait_r835_scp_then_chall_lunar_p3917.pid 2>/dev/null || true)
  if [[ "$opid" =~ ^[0-9]+$ ]] && kill -0 "$opid" 2>/dev/null; then
    echo "waiter already live pid=$opid"; exit 0
  fi
fi
nohup bash /root/mining_src/r835-chall/wait_r835_scp_then_chall_lunar_p3917.sh \
  >/root/logs/wait_r835_scp_then_chall_lunar_p3917.nohup 2>&1 &
echo $! >/root/logs/wait_r835_scp_then_chall_lunar_p3917.pid
echo "armed pid=$(cat /root/logs/wait_r835_scp_then_chall_lunar_p3917.pid)"
REMOTE
date -u +%Y-%m-%dT%H:%M:%SZ >"$DONE"
log "DONE — R835 merge→relay→lunar stamp path armed"
