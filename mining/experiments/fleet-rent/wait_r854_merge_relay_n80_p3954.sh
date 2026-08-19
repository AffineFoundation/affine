#!/usr/bin/env bash
# p3954: R854 already MERGE_DONE on gentle-shark → host-relay → lunar SIZE_OK → slot-claimer n80 vs reign36.
# Never pkill -f.
set -euo pipefail
ROOT=/home/const/subnet120/mining
LOGDIR=$ROOT/experiments/fleet-rent/logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/wait_r854_merge_relay_n80_p3954.log
: >"$LOG"
log() { echo "[p3954-r854-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -o BatchMode=yes)
SRC_HOST=86.38.182.67; SRC_PORT=20295
LUNAR_HOST=150.136.46.118; LUNAR_PORT=20299
RELAY=$ROOT/experiments/fleet-rent/host_relay_r854_to_lunar_p3954.sh
DONE=$LOGDIR/wait_r854_merge_relay_n80_p3954.done
[[ -f "$DONE" ]] && { log "already done"; exit 0; }

log "confirm R854 /root/logs/r854_merge.done"
st=$(ssh "${SSH_OPTS[@]}" -p "$SRC_PORT" "root@$SRC_HOST" \
  '[[ -f /root/logs/r854_merge.done && -f /tmp/r854_merged/config.json ]] && echo YES || echo NO' 2>/dev/null || echo ERR)
[[ "$st" == "YES" ]] || { log "FATAL merge not ready st=$st"; exit 1; }
log "merge.done seen — start relay"
nohup bash "$RELAY" >"$LOGDIR/host_relay_r854_to_lunar_p3954.outer.nohup" 2>&1 &
echo $! >"$LOGDIR/host_relay_r854_to_lunar_p3954.outer.pid"
log "relay pid=$(cat "$LOGDIR/host_relay_r854_to_lunar_p3954.outer.pid")"
wait "$(cat "$LOGDIR/host_relay_r854_to_lunar_p3954.outer.pid")"
[[ -f "$LOGDIR/host_relay_r854_to_lunar_p3954.done" ]] || { log "FATAL relay failed"; exit 1; }

ssh "${SSH_OPTS[@]}" -p "$LUNAR_PORT" "root@$LUNAR_HOST" "mkdir -p /root/mining_src/r854-chall /root/logs"
scp "${SSH_OPTS[@]}" -P "$LUNAR_PORT" \
  "$ROOT/experiments/fleet-rent/wait_r854_scp_then_chall_lunar_p3954.sh" \
  "$ROOT/experiments/fleet-rent/lean_chall_n80_lunar_slot_r854_p3954.sh" \
  "root@$LUNAR_HOST:/root/mining_src/r854-chall/"
ssh "${SSH_OPTS[@]}" -p "$LUNAR_PORT" "root@$LUNAR_HOST" bash -s <<'REMOTE'
set -euo pipefail
mkdir -p /root/mining_src/r854-chall /root/logs
chmod +x /root/mining_src/r854-chall/*.sh
if [[ -f /root/logs/wait_r854_scp_then_chall_lunar_p3954.pid ]]; then
  opid=$(cat /root/logs/wait_r854_scp_then_chall_lunar_p3954.pid 2>/dev/null || true)
  if [[ "$opid" =~ ^[0-9]+$ ]] && kill -0 "$opid" 2>/dev/null; then
    echo "waiter already live pid=$opid"; exit 0
  fi
fi
nohup bash /root/mining_src/r854-chall/wait_r854_scp_then_chall_lunar_p3954.sh \
  >/root/logs/wait_r854_scp_then_chall_lunar_p3954.nohup 2>&1 &
echo $! >/root/logs/wait_r854_scp_then_chall_lunar_p3954.pid
echo "armed pid=$(cat /root/logs/wait_r854_scp_then_chall_lunar_p3954.pid)"
REMOTE
date -u +%Y-%m-%dT%H:%M:%SZ >"$DONE"
log "DONE — R854 merge→relay→lunar stamp path armed"
