#!/usr/bin/env bash
# p3912: wait HYPO merge.done on brave → flock host-relay → crown → arm slot-claimer n80.
# Usage: wait_brave_merge_relay_n80_p3912.sh <r821|r822|r823|r824>
# Never pkill -f. Serializes relays via flock (no dual 66G pipes).
set -euo pipefail
hypo=${1:?hypo required}
case "$hypo" in r821|r822|r823|r824) ;; *) echo "bad hypo=$hypo"; exit 2 ;; esac

declare -A EDIR=(
  [r821]=r821-tammy-offline-dpo-hialpha-hirank-lobeta-midctx-megasuperextrasteps-ep4-ultralolr
  [r822]=r822-tammy-offline-dpo-hialpha-midrank-lobeta-shortctx-megasuperextrasteps-ep4-ultralolr
  [r823]=r823-tammy-offline-dpo-hialpha-hirank-lobeta-shortctx-megasuperextrasteps-ep4-ultralolr
  [r824]=r824-tammy-offline-dpo-hialpha-midrank-hibeta-shortctx-megasuperextrasteps-ep4-ultralolr
)

ROOT=/home/const/subnet120/mining
LOGDIR=$ROOT/experiments/fleet-rent/logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/wait_${hypo}_merge_relay_n80_p3912.log
: >"$LOG"
log() { echo "[p3912-$hypo-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -o BatchMode=yes)
BRAVE_HOST=18.118.83.97; BRAVE_PORT=40127
CROWN_HOST=95.133.253.90; CROWN_PORT=40099
RELAY=$ROOT/experiments/fleet-rent/host_relay_brave_to_crown_p3912.sh
LOCK=$LOGDIR/brave_crown_relay_p3912.lock
DONE=$LOGDIR/wait_${hypo}_merge_relay_n80_p3912.done
EXP=${EDIR[$hypo]}
[[ -f "$DONE" ]] && { log "already done"; exit 0; }

log "WAIT for /root/logs/${hypo}_merge.done on brave"
while true; do
  st=$(ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" \
    "[[ -f /root/logs/${hypo}_merge.done ]] && echo YES || echo NO" 2>/dev/null || echo ERR)
  [[ "$st" == "YES" ]] && break
  step=$(ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" \
    "grep -o '\"step\": [0-9]*' /root/logs/${hypo}_train.nohup 2>/dev/null | tail -1" 2>/dev/null || true)
  log "waiting merge.done (train $step)"
  sleep 45
done
log "merge.done seen — flock relay"

(
  flock -x 200
  log "relay lock acquired"
  if [[ -f "$LOGDIR/host_relay_${hypo}_brave_to_crown_p3912.done" ]]; then
    log "relay already done — skip pipes"
  else
    nohup bash "$RELAY" "$hypo" >"$LOGDIR/host_relay_${hypo}_brave_to_crown_p3912.outer.nohup" 2>&1 &
    rpid=$!
    echo "$rpid" >"$LOGDIR/host_relay_${hypo}_brave_to_crown_p3912.outer.pid"
    log "relay pid=$rpid"
    wait "$rpid"
    [[ -f "$LOGDIR/host_relay_${hypo}_brave_to_crown_p3912.done" ]] || { log "FATAL relay failed"; exit 1; }
  fi
) 200>"$LOCK"

log "arm crown slot-claimer for $hypo n80"
ssh "${SSH_OPTS[@]}" -p "$CROWN_PORT" "root@$CROWN_HOST" "mkdir -p /root/mining_src/${hypo}-chall /root/logs"
scp "${SSH_OPTS[@]}" -P "$CROWN_PORT" \
  "$ROOT/experiments/$EXP/lean_chall_n80_crown_slot_p3904.sh" \
  "$ROOT/experiments/$EXP/wait_${hypo}_scp_then_chall_p3912.sh" \
  "root@$CROWN_HOST:/root/mining_src/${hypo}-chall/"
ssh "${SSH_OPTS[@]}" -p "$CROWN_PORT" "root@$CROWN_HOST" bash -s <<REMOTE
set -euo pipefail
hypo=$hypo
mkdir -p /root/mining_src/${hypo}-chall /root/logs
chmod +x /root/mining_src/${hypo}-chall/*.sh
[[ -f /root/logs/${hypo}_chall_slot_claimed.p3912 ]] && exit 0
# idempotent: kill only our exact prior waiter pid if present
if [[ -f /root/logs/wait_${hypo}_scp_then_chall_p3912.pid ]]; then
  opid=\$(cat /root/logs/wait_${hypo}_scp_then_chall_p3912.pid 2>/dev/null || true)
  if [[ "\$opid" =~ ^[0-9]+\$ ]] && kill -0 "\$opid" 2>/dev/null; then
    echo "waiter already live pid=\$opid"; exit 0
  fi
fi
nohup bash /root/mining_src/${hypo}-chall/wait_${hypo}_scp_then_chall_p3912.sh \
  >/root/logs/wait_${hypo}_scp_then_chall_p3912.nohup 2>&1 &
echo \$! >/root/logs/wait_${hypo}_scp_then_chall_p3912.pid
echo "armed pid=\$(cat /root/logs/wait_${hypo}_scp_then_chall_p3912.pid)"
REMOTE
date -u +%Y-%m-%dT%H:%M:%SZ >"$DONE"
log "DONE — $hypo merge→relay→crown slot-waiter armed"
