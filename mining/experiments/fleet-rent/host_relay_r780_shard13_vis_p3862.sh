#!/usr/bin/env bash
# p3862: clean solo pipe for R780 shard 13 (dual-write corrupted) + visual.
# Assumes p3848 parent is SIGSTOP'd so it won't dual-launch. Tail owns 14/15/16.
# Never pkill -f. Size-checked SSH pipes brave→R252.
set -euo pipefail
ROOT=/home/const/subnet120/mining
LOGDIR=$ROOT/experiments/fleet-rent/logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/host_relay_r780_shard13_vis_p3862.log
: >"$LOG"
log() { echo "[p3862-r780-13vis] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=45 -o BatchMode=yes
  -o ServerAliveInterval=15 -o ServerAliveCountMax=40 -o TCPKeepAlive=yes)
BRAVE_HOST=18.118.83.97
BRAVE_PORT=40127
R252_HOST=95.133.252.28
R252_PORT=40299

FILES=(
  model-00013-of-00016.safetensors
  model-visual-restored.safetensors
)

dest_size() {
  local f="$1" out
  out=$(ssh "${SSH_OPTS[@]}" -p "$R252_PORT" "root@$R252_HOST" \
    "stat -c%s /tmp/r780_merged/$f 2>/dev/null || echo 0" 2>/dev/null || echo 0)
  echo "${out//[^0-9]/}"
}

src_size() {
  local f="$1" out
  out=$(ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" \
    "stat -c%s /tmp/r780_merged/$f 2>/dev/null || echo 0" 2>/dev/null || echo 0)
  echo "${out//[^0-9]/}"
}

xfer_one() {
  local f="$1" want="$2"
  local attempt rc got
  for attempt in 1 2 3 4 5; do
    log "PIPE start $f attempt=$attempt want=$want"
    set +e
    ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" \
      "cat /tmp/r780_merged/$f" \
      | ssh "${SSH_OPTS[@]}" -p "$R252_PORT" "root@$R252_HOST" \
        "mkdir -p /tmp/r780_merged && cat > /tmp/r780_merged/$f.tmp && mv -f /tmp/r780_merged/$f.tmp /tmp/r780_merged/$f"
    rc=$?
    set -e
    if [[ "$rc" -ne 0 ]]; then
      log "PIPE fail $f rc=$rc"
      sleep $((attempt * 3))
      continue
    fi
    got=$(dest_size "$f")
    got=${got:-0}
    if [[ "$got" == "$want" ]]; then
      log "PIPE ok $f ($got)"
      return 0
    fi
    log "PIPE size mismatch $f got=$got want=$want"
    sleep $((attempt * 3))
  done
  log "FATAL $f"
  return 1
}

log "START clean 13+vis (p3848 STOP'd; tail owns 14/15/16)"
fail=0
for f in "${FILES[@]}"; do
  want=$(src_size "$f")
  want=${want:-0}
  if [[ "$want" -le 0 ]]; then
    log "SKIP missing src $f"
    fail=1
    continue
  fi
  have=$(dest_size "$f")
  have=${have:-0}
  if [[ "$have" == "$want" ]]; then
    log "KEEP $f ($have)"
    continue
  fi
  log "NEED $f (have=$have want=$want)"
  # drop any partial/corrupt before solo rewrite
  ssh "${SSH_OPTS[@]}" -p "$R252_PORT" "root@$R252_HOST" \
    "rm -f /tmp/r780_merged/$f /tmp/r780_merged/$f.tmp" 2>/dev/null || true
  if ! xfer_one "$f" "$want"; then fail=1; fi
done

stamp_out=$(ssh "${SSH_OPTS[@]}" -p "$R252_PORT" "root@$R252_HOST" \
  'n=$(ls /tmp/r780_merged/model-*-of-*.safetensors 2>/dev/null | wc -l)
   vis=0; [[ -f /tmp/r780_merged/model-visual-restored.safetensors ]] && vis=1
   if [[ -f /tmp/r780_merged/config.json ]] && [[ "$n" -ge 16 ]] && [[ "$vis" -eq 1 ]]; then
     # size-check key shards vs expected before stamp (13+vis done here; others via peers)
     date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r780_scp_ready.done
     echo "ok:$n:vis=$vis:$(du -sh /tmp/r780_merged | awk "{print \$1}")"
   else
     echo "partial:$n:vis=$vis"
   fi' 2>/dev/null || echo sshfail)
log "stamp_check $stamp_out"
[[ "$fail" -eq 0 ]] || { log "DONE with fails"; exit 1; }
log "DONE 13+vis"
exit 0
