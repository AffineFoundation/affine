#!/usr/bin/env bash
# p3867: R781 finish — size-verify ALL 16 shards + visual vs brave, then kill
# STOP'd p3848 parent (never CONT), then stamp r781_scp_ready.done.
# Count-only stamps (mid/tail/parent) can license truncated shards (p3863).
# Never pkill -f. Leave R802 TRAIN 6,7 and teacher/king alone.
set -euo pipefail
ROOT=/home/const/subnet120/mining
LOGDIR=$ROOT/experiments/fleet-rent/logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/host_relay_r781_finish_sizeverify_stamp_p3867.log
: >"$LOG"
log() { echo "[p3867-r781-fin] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=45 -o BatchMode=yes
  -o ServerAliveInterval=15 -o ServerAliveCountMax=40 -o TCPKeepAlive=yes)
BRAVE_HOST=18.118.83.97
BRAVE_PORT=40127
R252_HOST=95.133.252.28
R252_PORT=40299
HUID=gentle-wolf-8c
PARENT_PID=${R781_STOP_PARENT:-3405447}

FILES=(
  model-00001-of-00016.safetensors
  model-00002-of-00016.safetensors
  model-00003-of-00016.safetensors
  model-00004-of-00016.safetensors
  model-00005-of-00016.safetensors
  model-00006-of-00016.safetensors
  model-00007-of-00016.safetensors
  model-00008-of-00016.safetensors
  model-00009-of-00016.safetensors
  model-00010-of-00016.safetensors
  model-00011-of-00016.safetensors
  model-00012-of-00016.safetensors
  model-00013-of-00016.safetensors
  model-00014-of-00016.safetensors
  model-00015-of-00016.safetensors
  model-00016-of-00016.safetensors
  model-visual-restored.safetensors
)

dest_size() {
  local f="$1" out
  out=$(ssh "${SSH_OPTS[@]}" -p "$R252_PORT" "root@$R252_HOST" \
    "stat -c%s /tmp/r781_merged/$f 2>/dev/null || echo 0" 2>/dev/null || echo 0)
  echo "${out//[^0-9]/}"
}

src_size() {
  local f="$1" out
  out=$(ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" \
    "stat -c%s /tmp/r781_merged/$f 2>/dev/null || echo 0" 2>/dev/null || echo 0)
  echo "${out//[^0-9]/}"
}

tmp_count() {
  local out
  out=$(ssh "${SSH_OPTS[@]}" -p "$R252_PORT" "root@$R252_HOST" \
    'ls /tmp/r781_merged/*.tmp 2>/dev/null | wc -l' 2>/dev/null || echo 99)
  echo "${out//[^0-9]/}"
}

kill_stop_parent() {
  local pid=$1
  [[ "$pid" =~ ^[0-9]+$ ]] || return 0
  if ! kill -0 "$pid" 2>/dev/null; then
    log "parent $pid already gone"
    return 0
  fi
  local st
  st=$(awk '/^State:/{print $2}' /proc/"$pid"/status 2>/dev/null || echo '?')
  log "kill STOP parent pid=$pid state=$st (never CONT)"
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 40); do
    kill -0 "$pid" 2>/dev/null || { log "parent $pid exited"; return 0; }
    sleep 1
  done
  kill -9 "$pid" 2>/dev/null || true
  log "parent $pid kill -9 sent"
}

stamp_ready() {
  local out
  out=$(ssh "${SSH_OPTS[@]}" -p "$R252_PORT" "root@$R252_HOST" \
    'mkdir -p /root/logs
     date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r781_scp_ready.done
     n=$(ls /tmp/r781_merged/model-*-of-*.safetensors 2>/dev/null | wc -l)
     vis=0; [[ -f /tmp/r781_merged/model-visual-restored.safetensors ]] && vis=1
     echo "ok:$n:vis=$vis:$(du -sh /tmp/r781_merged | awk "{print \$1}")"' \
    2>/dev/null || echo sshfail)
  echo "$out"
}

log "armed: size-verify 16+vis then kill parent=$PARENT_PID then stamp (timeout ~3h)"

for i in $(seq 1 720); do
  bad=0
  miss=()
  for f in "${FILES[@]}"; do
    want=$(src_size "$f")
    want=${want:-0}
    if [[ "$want" -le 0 ]]; then
      miss+=("$f:nosrc")
      bad=1
      continue
    fi
    have=$(dest_size "$f")
    have=${have:-0}
    if [[ "$have" != "$want" ]]; then
      miss+=("$f:$have/$want")
      bad=1
    fi
  done
  tmps=$(tmp_count)
  tmps=${tmps:-99}

  # also require config.json
  cfg=$(ssh "${SSH_OPTS[@]}" -p "$R252_PORT" "root@$R252_HOST" \
    'test -f /tmp/r781_merged/config.json && echo 1 || echo 0' 2>/dev/null || echo 0)

  if [[ "$bad" -eq 0 && "$tmps" -eq 0 && "$cfg" == "1" ]]; then
    log "SIZE_OK poll=$i all 17 files match src; tmp=0 cfg=1"
    kill_stop_parent "$PARENT_PID"
    # reap any leftover STOP'd child bash wrappers of the parent (exact PIDs only)
    for cp in $(pgrep -P "$PARENT_PID" 2>/dev/null || true); do
      kill_stop_parent "$cp"
    done
    ready=$(stamp_ready)
    log "SCP_READY $ready"
    # confirm stamp readable
    conf=$(ssh "${SSH_OPTS[@]}" -p "$R252_PORT" "root@$R252_HOST" \
      'cat /root/logs/r781_scp_ready.done 2>/dev/null; ls /tmp/r781_merged/model-*-of-*.safetensors 2>/dev/null | wc -l' \
      2>/dev/null || echo fail)
    log "confirm $conf"
    log "DONE — pod waiter 463456 may launch lean n80 :8002"
    exit 0
  fi

  if [[ $((i % 4)) -eq 0 ]]; then
    log "poll i=$i bad=$bad tmp=$tmps cfg=$cfg miss=${miss[*]:-none}"
  fi
  sleep 15
done

log "FATAL timeout waiting size-ok"
exit 1
