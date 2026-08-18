#!/usr/bin/env bash
# p3847: R762 REFUTE → R767 solo tar stall (~4G/66G) → kill by PID,
# resume with parallel size-checked host pipes (4 concurrent).
# Never pkill -f. Leave R793 TRAIN 6,7 alone. Leave R768 waiter armed.
set -euo pipefail
ROOT=/home/const/subnet120/mining
LOGDIR=$ROOT/experiments/fleet-rent/logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/host_relay_r767_parallel_resume_p3847.log
OLDLOG=$LOGDIR/host_relay_r767_resume_after_r762_p3836.log
: >"$LOG"
log() { echo "[p3847-r767] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=45 -o BatchMode=yes
  -o ServerAliveInterval=15 -o ServerAliveCountMax=40 -o TCPKeepAlive=yes)
BRAVE_HOST=18.118.83.97
BRAVE_PORT=40127
R252_HOST=95.133.252.28
R252_PORT=40299
HUID=gentle-wolf-8c
NPARA=4

log "START parallel resume R767 brave→R252 (supersede slow tar after R762 REFUTE)"

kill_old_tar() {
  local pids=("$@")
  if [[ ${#pids[@]} -eq 0 ]]; then
    while read -r pid; do
      [[ -n "$pid" ]] && pids+=("$pid")
    done < <(pgrep -f 'host_relay_r767_resume_after_r762_p3836\.sh' 2>/dev/null || true)
    while read -r pid; do
      [[ -n "$pid" ]] && pids+=("$pid")
    done < <(pgrep -f 'tar cf - r767_merged' 2>/dev/null || true)
    while read -r pid; do
      [[ -n "$pid" ]] && pids+=("$pid")
    done < <(pgrep -f 'r767_scp_ready\.done' 2>/dev/null || true)
  fi
  local uniq=() seen=""
  for pid in "${pids[@]:-}"; do
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    case " $seen " in
      *" $pid "*) continue ;;
    esac
    seen+=" $pid"
    uniq+=("$pid")
  done
  pids=("${uniq[@]:-}")
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
  # remote tar on brave (exact pid from ps if present)
  ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" \
    'p=$(pgrep -n -f "^tar cf - r767_merged$" || true); if [[ -n "$p" ]]; then kill "$p" 2>/dev/null || true; sleep 1; kill -9 "$p" 2>/dev/null || true; echo killed_remote_tar=$p; else echo no_remote_tar; fi' \
    || log "WARN brave tar kill soft-fail"
  echo "[p3836-r767] $(date -u +%Y-%m-%dT%H:%M:%SZ) SUPERSEDED by p3847 parallel resume (tar killed)" >>"$OLDLOG"
  log "old tar stopped"
}

kill_old_tar "$@"

mapfile -t SRC_LINES < <(ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" \
  'cd /tmp/r767_merged && for f in model-*-of-*.safetensors model-visual-restored.safetensors config.json tokenizer.json tokenizer_config.json generation_config.json merge_meta.json model.safetensors.index.json chat_template.jinja preprocessor_config.json processor_config.json video_preprocessor_config.json README.md; do
     [[ -f "$f" ]] && printf "%s %s\n" "$f" "$(stat -c%s "$f")"
   done')
log "brave files=${#SRC_LINES[@]}"
[[ ${#SRC_LINES[@]} -ge 17 ]] || { log "FATAL source map short n=${#SRC_LINES[@]}"; exit 1; }

clear_ready() {
  if timeout 55 lium exec "$HUID" 'rm -f /root/logs/r767_scp_ready.done; mkdir -p /tmp/r767_merged /root/logs; echo cleared' 2>/dev/null | grep -q cleared; then
    return 0
  fi
  ssh "${SSH_OPTS[@]}" -p "$R252_PORT" "root@$R252_HOST" \
    'mkdir -p /tmp/r767_merged /root/logs; rm -f /root/logs/r767_scp_ready.done'
}
clear_ready || log "WARN clear_ready soft-fail (continue)"

dest_size() {
  local f="$1" out
  out=$(timeout 55 lium exec "$HUID" "stat -c%s /tmp/r767_merged/$f 2>/dev/null || echo 0" 2>/dev/null \
    | grep -E '^[0-9]+$' | tail -1 || true)
  if [[ -n "$out" ]]; then echo "$out"; return 0; fi
  out=$(ssh "${SSH_OPTS[@]}" -p "$R252_PORT" "root@$R252_HOST" \
    "stat -c%s /tmp/r767_merged/$f 2>/dev/null || echo 0" 2>/dev/null || echo 0)
  echo "${out//[^0-9]/}"
}

rm_dest() {
  local f="$1"
  timeout 55 lium exec "$HUID" "rm -f /tmp/r767_merged/$f" >/dev/null 2>&1 || true
  ssh "${SSH_OPTS[@]}" -p "$R252_PORT" "root@$R252_HOST" "rm -f /tmp/r767_merged/$f" 2>/dev/null || true
}

need_list=()
for line in "${SRC_LINES[@]}"; do
  f=${line%% *}
  want=${line##* }
  have=$(dest_size "$f")
  have=${have:-0}
  if [[ "$have" == "$want" ]]; then
    log "KEEP $f ($have)"
  else
    log "NEED $f (have=$have want=$want)"
    need_list+=("$f:$want")
    rm_dest "$f"
  fi
done
log "need_count=${#need_list[@]}"

xfer_one() {
  local f="$1" want="$2"
  local attempt rc got
  for attempt in 1 2 3 4 5; do
    log "PIPE start $f attempt=$attempt"
    set +e
    ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" \
      "cat /tmp/r767_merged/$f" \
      | ssh "${SSH_OPTS[@]}" -p "$R252_PORT" "root@$R252_HOST" \
        "mkdir -p /tmp/r767_merged && cat > /tmp/r767_merged/$f.tmp && mv -f /tmp/r767_merged/$f.tmp /tmp/r767_merged/$f"
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
  log "FATAL $f after retries"
  return 1
}

fail=0
active=0
pids=()
for item in "${need_list[@]}"; do
  f=${item%%:*}
  want=${item##*:}
  while [[ "$active" -ge "$NPARA" ]]; do
    if ! wait -n; then fail=1; fi
    active=0
    live=()
    for p in "${pids[@]:-}"; do
      if kill -0 "$p" 2>/dev/null; then
        live+=("$p")
        active=$((active + 1))
      fi
    done
    pids=("${live[@]:-}")
  done
  xfer_one "$f" "$want" &
  pids+=("$!")
  active=$((active + 1))
done
for p in "${pids[@]:-}"; do
  if ! wait "$p"; then fail=1; fi
done
[[ "$fail" -eq 0 ]] || { log "FATAL some transfers failed"; exit 4; }

verify_and_stamp() {
  local out
  out=$(timeout 90 lium exec "$HUID" 'n=$(ls /tmp/r767_merged/model-*-of-*.safetensors 2>/dev/null | wc -l)
    vis=0; [[ -f /tmp/r767_merged/model-visual-restored.safetensors ]] && vis=1
    if [[ -f /tmp/r767_merged/config.json ]] && [[ "$n" -ge 16 ]]; then
      date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r767_scp_ready.done
      echo "ok:$n:vis=$vis:$(du -sh /tmp/r767_merged | awk "{print \$1}")"
    else
      echo "bad:$n:vis=$vis"
    fi' 2>/dev/null | grep -E '^(ok|bad):' | tail -1 || true)
  if [[ "$out" == ok:* ]]; then
    echo "$out"
    return 0
  fi
  out=$(ssh "${SSH_OPTS[@]}" -p "$R252_PORT" "root@$R252_HOST" \
    'n=$(ls /tmp/r767_merged/model-*-of-*.safetensors 2>/dev/null | wc -l)
     vis=0; [[ -f /tmp/r767_merged/model-visual-restored.safetensors ]] && vis=1
     if [[ -f /tmp/r767_merged/config.json ]] && [[ "$n" -ge 16 ]]; then
       date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r767_scp_ready.done
       echo "ok:$n:vis=$vis:$(du -sh /tmp/r767_merged | awk "{print \$1}")"
     else
       echo "bad:$n:vis=$vis"
     fi' 2>/dev/null || true)
  echo "$out"
  [[ "$out" == ok:* ]]
}

ready=""
for i in 1 2 3 4 5; do
  if ready=$(verify_and_stamp); then
    log "SCP_READY $ready"
    echo "[p3836-r767] $(date -u +%Y-%m-%dT%H:%M:%SZ) DONE relay (via p3847 parallel) $ready" >>"$OLDLOG"
    log "DONE parallel resume — waiter may launch R767 n80"
    exit 0
  fi
  log "verify attempt=$i got=$ready"
  sleep 5
done
log "FATAL verify failed"
exit 5
