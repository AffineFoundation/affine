#!/usr/bin/env bash
# p3881: SIGSTOP p3879 R800 parent; accel free shards 09–16+vis+meta while
# live pipes finish 05–08; SIZE_OK → kill STOP'd parent → stamp r800_scp_ready.done
# + write p3879.done so R801 waiter proceeds. Never pkill -f. Never CONT parent.
set -euo pipefail
ROOT=/home/const/subnet120/mining
LOGDIR=$ROOT/experiments/fleet-rent/logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/host_relay_r800_tail_meta_accel_p3881.log
: >"$LOG"
log() { echo "[p3881-r800-tail] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=45 -o BatchMode=yes
  -o ServerAliveInterval=15 -o ServerAliveCountMax=40 -o TCPKeepAlive=yes)
BRAVE_HOST=18.118.83.97; BRAVE_PORT=40127
CROWN_HOST=95.133.253.90; CROWN_PORT=40099
NPARA=6
PARENT_P3879=3852020
hypo=r800

CAND=(
  model-00009-of-00016.safetensors
  model-00010-of-00016.safetensors
  model-00011-of-00016.safetensors
  model-00012-of-00016.safetensors
  model-00013-of-00016.safetensors
  model-00014-of-00016.safetensors
  model-00015-of-00016.safetensors
  model-00016-of-00016.safetensors
  model-visual-restored.safetensors
  config.json
  tokenizer.json
  tokenizer_config.json
  generation_config.json
  merge_meta.json
  model.safetensors.index.json
  chat_template.jinja
  preprocessor_config.json
  processor_config.json
  video_preprocessor_config.json
)

src_size() {
  local f=$1 out
  out=$(ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" \
    "stat -c%s /tmp/${hypo}_merged/$f 2>/dev/null || echo 0" 2>/dev/null || echo 0)
  echo "${out//[^0-9]/}"
}

dest_size() {
  local f=$1 out
  out=$(ssh "${SSH_OPTS[@]}" -p "$CROWN_PORT" "root@$CROWN_HOST" \
    "stat -c%s /tmp/${hypo}_merged/$f 2>/dev/null || echo 0" 2>/dev/null || echo 0)
  echo "${out//[^0-9]/}"
}

busy_tmp() {
  local f=$1
  ssh "${SSH_OPTS[@]}" -p "$CROWN_PORT" "root@$CROWN_HOST" \
    "test -f /tmp/${hypo}_merged/$f.tmp && echo yes || echo no" 2>/dev/null || echo yes
}

xfer_one() {
  local f=$1 want=$2
  local attempt rc got
  for attempt in 1 2 3 4 5; do
    if [[ "$(busy_tmp "$f")" == yes ]]; then
      log "BUSY abort $f (live .tmp)"
      return 1
    fi
    log "PIPE $f attempt=$attempt want=$want"
    set +e
    ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" \
      "cat /tmp/${hypo}_merged/$f" \
      | ssh "${SSH_OPTS[@]}" -p "$CROWN_PORT" "root@$CROWN_HOST" \
        "mkdir -p /tmp/${hypo}_merged && cat > /tmp/${hypo}_merged/$f.tmp && mv -f /tmp/${hypo}_merged/$f.tmp /tmp/${hypo}_merged/$f"
    rc=$?
    set -e
    if [[ "$rc" -ne 0 ]]; then
      log "PIPE fail $f rc=$rc"
      sleep $((attempt * 3))
      continue
    fi
    got=$(dest_size "$f"); got=${got:-0}
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

if kill -0 "$PARENT_P3879" 2>/dev/null; then
  kill -STOP "$PARENT_P3879" 2>/dev/null || true
  log "SIGSTOP p3879 parent pid=$PARENT_P3879 (live children keep running 05-08)"
else
  log "WARN p3879 parent $PARENT_P3879 already gone"
fi

log "START tail+meta accel; skip busy .tmp / size-matched"
need_list=()
for f in "${CAND[@]}"; do
  want=$(src_size "$f"); want=${want:-0}
  if [[ "$want" -le 0 ]]; then
    log "SKIP nosrc $f"
    continue
  fi
  have=$(dest_size "$f"); have=${have:-0}
  if [[ "$have" == "$want" ]]; then
    log "KEEP $f ($have)"
    continue
  fi
  if [[ "$(busy_tmp "$f")" == yes ]]; then
    log "BUSY skip $f"
    continue
  fi
  log "NEED $f (have=$have want=$want)"
  need_list+=("$f:$want")
done
log "need_count=${#need_list[@]}"

fail=0; active=0; pids=()
for item in "${need_list[@]}"; do
  f=${item%%:*}; want=${item##*:}
  while [[ "$active" -ge "$NPARA" ]]; do
    if ! wait -n; then fail=1; fi
    active=0; live=()
    for p in "${pids[@]:-}"; do
      if kill -0 "$p" 2>/dev/null; then live+=("$p"); active=$((active + 1)); fi
    done
    pids=("${live[@]:-}")
  done
  xfer_one "$f" "$want" &
  pids+=("$!"); active=$((active + 1))
done
for p in "${pids[@]:-}"; do
  if ! wait "$p"; then fail=1; fi
done
log "tail+meta wave done fail=$fail"

log "poll until R800 size-complete (incl. live 05-08 pipes)"
FILES_FULL=(
  model-00001-of-00016.safetensors model-00002-of-00016.safetensors
  model-00003-of-00016.safetensors model-00004-of-00016.safetensors
  model-00005-of-00016.safetensors model-00006-of-00016.safetensors
  model-00007-of-00016.safetensors model-00008-of-00016.safetensors
  model-00009-of-00016.safetensors model-00010-of-00016.safetensors
  model-00011-of-00016.safetensors model-00012-of-00016.safetensors
  model-00013-of-00016.safetensors model-00014-of-00016.safetensors
  model-00015-of-00016.safetensors model-00016-of-00016.safetensors
  model-visual-restored.safetensors config.json
)
ok=0
for i in $(seq 1 600); do
  bad=0; tmps=0
  tmps=$(ssh "${SSH_OPTS[@]}" -p "$CROWN_PORT" "root@$CROWN_HOST" \
    "ls /tmp/${hypo}_merged/*.tmp 2>/dev/null | wc -l" 2>/dev/null || echo 99)
  tmps=${tmps//[^0-9]/}; tmps=${tmps:-99}
  for f in "${FILES_FULL[@]}"; do
    want=$(src_size "$f"); want=${want:-0}
    have=$(dest_size "$f"); have=${have:-0}
    if [[ "$want" -le 0 || "$have" != "$want" ]]; then
      bad=1
      if [[ "$want" -gt 0 && "$have" != "$want" && "$(busy_tmp "$f")" != yes ]]; then
        log "RECOVER $f have=$have want=$want"
        xfer_one "$f" "$want" || true
      fi
    fi
  done
  if [[ "$bad" -eq 0 && "$tmps" -eq 0 ]]; then
    log "R800 SIZE_OK iter=$i"
    ok=1
    break
  fi
  (( i % 12 == 0 )) && log "R800 wait iter=$i bad=$bad tmps=$tmps"
  sleep 10
done
[[ "$ok" -eq 1 ]] || { log "FATAL never SIZE_OK"; exit 1; }

# Kill STOP'd parent (never CONT) — would resume and dual-write
if kill -0 "$PARENT_P3879" 2>/dev/null; then
  kill -9 "$PARENT_P3879" 2>/dev/null || true
  log "killed STOP'd p3879 parent $PARENT_P3879"
fi
# Reap orphan xfer bash children of dead parent with no live ssh
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  if ! pgrep -P "$pid" -a 2>/dev/null | grep -q ssh; then
    kill -9 "$pid" 2>/dev/null || true
    log "reap orphan xfer bash $pid"
  fi
done < <(pgrep -f 'host_relay_r800_fast_p3879\.sh' || true)

# Copy remaining small meta if missing (tokenizer etc already in CAND)
for f in tokenizer.json tokenizer_config.json generation_config.json merge_meta.json \
         model.safetensors.index.json chat_template.jinja preprocessor_config.json \
         processor_config.json video_preprocessor_config.json; do
  want=$(src_size "$f"); want=${want:-0}
  [[ "$want" -gt 0 ]] || continue
  have=$(dest_size "$f"); have=${have:-0}
  [[ "$have" == "$want" ]] && continue
  xfer_one "$f" "$want" || log "WARN meta miss $f"
done

stamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
ssh "${SSH_OPTS[@]}" -p "$CROWN_PORT" "root@$CROWN_HOST" \
  "echo '$stamp' > /root/logs/${hypo}_scp_ready.done && echo stamped:\$(cat /root/logs/${hypo}_scp_ready.done):\$(ls /tmp/${hypo}_merged/model-*-of-*.safetensors | wc -l)"
log "STAMPED r800_scp_ready.done=$stamp"

# Unblock R801 waiter that polls p3879.done
date -u +%Y-%m-%dT%H:%M:%SZ >"$LOGDIR/host_relay_r800_fast_p3879.done"
log "wrote host_relay_r800_fast_p3879.done (R801 may proceed)"

date -u +%Y-%m-%dT%H:%M:%SZ >"$LOGDIR/host_relay_r800_tail_meta_accel_p3881.done"
log "DONE p3881 R800 tail+meta accel + stamp"
exit 0
