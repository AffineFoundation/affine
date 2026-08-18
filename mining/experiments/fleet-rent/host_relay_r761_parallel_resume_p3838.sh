#!/usr/bin/env bash
# p3838: R761 single-stream tar ~23G/66G in 45m (partial shard-07) → kill by PID
# and resume with parallel size-checked host pipes (4 concurrent). Never pkill -f.
# Leave R782 TRAIN 6,7 + n80 waiter pid371826 alone. R762+ waiters still gate on
# /root/logs/r761_scp_ready.done + 16 shards.
set -euo pipefail
ROOT=/home/const/subnet120/mining
LOGDIR=$ROOT/experiments/fleet-rent/logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/host_relay_r761_parallel_resume_p3838.log
OLDLOG=$LOGDIR/host_relay_r761_brave_to_r252_p3830.log
: >"$LOG"
log() { echo "[p3838-r761] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=45 -o BatchMode=yes
  -o ServerAliveInterval=15 -o ServerAliveCountMax=40 -o TCPKeepAlive=yes)
BRAVE_HOST=18.118.83.97
BRAVE_PORT=40127
R252_HOST=95.133.252.28
R252_PORT=40299
HUID=gentle-wolf-8c
NPARA=4

log "START parallel resume R761 brave→R252 (supersede slow tar)"

# 1) Kill old tar-pipe PIDs only (exact PIDs from argv or discovered)
kill_old_tar() {
  local pids=("$@")
  if [[ ${#pids[@]} -eq 0 ]]; then
    # discover: parent host_relay_r761_brave_to_r252_p3830 + its two ssh children
    while read -r pid; do
      [[ -n "$pid" ]] && pids+=("$pid")
    done < <(pgrep -P 1 -f 'host_relay_r761_brave_to_r252_p3830\.sh' 2>/dev/null || true)
    for parent in "${pids[@]:-}"; do
      while read -r c; do
        [[ -n "$c" ]] && pids+=("$c")
      done < <(pgrep -P "$parent" 2>/dev/null || true)
    done
  fi
  if [[ ${#pids[@]} -eq 0 ]]; then
    log "no old tar PIDs found (already gone?)"
    return 0
  fi
  log "killing exact PIDs: ${pids[*]}"
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  sleep 2
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null || true
    fi
  done
  echo "[p3830-r761] $(date -u +%Y-%m-%dT%H:%M:%SZ) SUPERSEDED by p3838 parallel resume (tar killed)" >>"$OLDLOG"
  log "old tar stopped"
}

# Prefer explicit PIDs passed as args (safer than pgrep)
kill_old_tar "$@"

# 2) Source size map from brave
mapfile -t SRC_LINES < <(ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" \
  'cd /tmp/r761_merged && for f in model-*-of-*.safetensors config.json tokenizer.json tokenizer_config.json generation_config.json merge_meta.json model.safetensors.index.json chat_template.jinja preprocessor_config.json processor_config.json video_preprocessor_config.json; do
     [[ -f "$f" ]] && printf "%s %s\n" "$f" "$(stat -c%s "$f")"
   done')
log "brave files=${#SRC_LINES[@]}"
[[ ${#SRC_LINES[@]} -ge 17 ]] || { log "FATAL source map short"; exit 1; }

# 3) Ensure dest dir + drop incomplete wrong-size shards
ssh "${SSH_OPTS[@]}" -p "$R252_PORT" "root@$R252_HOST" \
  'mkdir -p /tmp/r761_merged /root/logs; rm -f /root/logs/r761_scp_ready.done'

need_list=()
for line in "${SRC_LINES[@]}"; do
  f=${line%% *}
  want=${line##* }
  have=$(ssh "${SSH_OPTS[@]}" -p "$R252_PORT" "root@$R252_HOST" \
    "stat -c%s /tmp/r761_merged/$f 2>/dev/null || echo 0" || echo 0)
  have=${have//[^0-9]/}
  have=${have:-0}
  if [[ "$have" == "$want" ]]; then
    log "KEEP $f ($have)"
  else
    log "NEED $f (have=$have want=$want)"
    need_list+=("$f:$want")
    # remove wrong/partial before rewrite
    ssh "${SSH_OPTS[@]}" -p "$R252_PORT" "root@$R252_HOST" \
      "rm -f /tmp/r761_merged/$f" || true
  fi
done
log "need_count=${#need_list[@]}"

xfer_one() {
  local f="$1" want="$2"
  local attempt rc
  for attempt in 1 2 3 4 5; do
    log "PIPE start $f attempt=$attempt"
    set +e
    ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" \
      "cat /tmp/r761_merged/$f" \
      | ssh "${SSH_OPTS[@]}" -p "$R252_PORT" "root@$R252_HOST" \
        "cat > /tmp/r761_merged/$f.tmp && mv -f /tmp/r761_merged/$f.tmp /tmp/r761_merged/$f"
    rc=$?
    set -e
    if [[ "$rc" -ne 0 ]]; then
      log "PIPE fail $f rc=$rc"
      sleep $((attempt * 3))
      continue
    fi
    got=$(ssh "${SSH_OPTS[@]}" -p "$R252_PORT" "root@$R252_HOST" \
      "stat -c%s /tmp/r761_merged/$f 2>/dev/null || echo 0" || echo 0)
    got=${got//[^0-9]/}
    if [[ "$got" == "$want" ]]; then
      log "PIPE ok $f ($got)"
      return 0
    fi
    log "PIPE size mismatch $f got=$got want=$want"
    sleep $((attempt * 3))
  done
  log "FATAL $f after retries"
  return 1
}

# 4) Parallel transfers
fail=0
active=0
pids=()
names=()
for item in "${need_list[@]}"; do
  f=${item%%:*}
  want=${item##*:}
  while [[ "$active" -ge "$NPARA" ]]; do
    # wait for any
    if ! wait -n; then fail=1; fi
    active=0
    for p in "${pids[@]:-}"; do
      if kill -0 "$p" 2>/dev/null; then active=$((active + 1)); fi
    done
    # rebuild live pid list
    live=()
    for p in "${pids[@]:-}"; do
      if kill -0 "$p" 2>/dev/null; then live+=("$p"); fi
    done
    pids=("${live[@]:-}")
  done
  xfer_one "$f" "$want" &
  pids+=("$!")
  names+=("$f")
  active=$((active + 1))
done
# drain
for p in "${pids[@]:-}"; do
  if ! wait "$p"; then fail=1; fi
done
[[ "$fail" -eq 0 ]] || { log "FATAL some transfers failed"; exit 4; }

# 5) Verify + stamp SCP_READY (prefer lium; SSH fallback)
verify_and_stamp() {
  local out
  out=$(timeout 90 lium exec "$HUID" 'n=$(ls /tmp/r761_merged/model-*-of-*.safetensors 2>/dev/null | wc -l)
    if [[ -f /tmp/r761_merged/config.json ]] && [[ "$n" -ge 16 ]]; then
      date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r761_scp_ready.done
      echo "ok:$n:$(du -sh /tmp/r761_merged | awk "{print \$1}")"
    else
      echo "bad:$n"
    fi' 2>/dev/null | grep -E '^(ok|bad):' | tail -1 || true)
  if [[ "$out" == ok:* ]]; then
    echo "$out"
    return 0
  fi
  out=$(ssh "${SSH_OPTS[@]}" -p "$R252_PORT" "root@$R252_HOST" \
    'n=$(ls /tmp/r761_merged/model-*-of-*.safetensors 2>/dev/null | wc -l)
     if [[ -f /tmp/r761_merged/config.json ]] && [[ "$n" -ge 16 ]]; then
       date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r761_scp_ready.done
       echo "ok:$n:$(du -sh /tmp/r761_merged | awk "{print \$1}")"
     else
       echo "bad:$n"
     fi' 2>/dev/null || true)
  echo "$out"
  [[ "$out" == ok:* ]]
}

ready=""
for i in 1 2 3 4 5; do
  if ready=$(verify_and_stamp); then
    log "SCP_READY $ready"
    echo "[p3830-r761] $(date -u +%Y-%m-%dT%H:%M:%SZ) DONE relay (via p3838 parallel) $ready" >>"$OLDLOG"
    log "DONE parallel resume — waiter may launch R761 n80"
    exit 0
  fi
  log "verify attempt=$i got=$ready"
  sleep 5
done
log "FATAL verify failed"
exit 5
