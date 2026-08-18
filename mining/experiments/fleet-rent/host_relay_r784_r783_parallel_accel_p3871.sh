#!/usr/bin/env bash
# p3871: kill slow sequential tar R784 brave→crown; parallel×4 size-checked
# pipes for R784 then R783. Keep waiters. Never pkill -f. Leave R800/R801.
set -euo pipefail
ROOT=/home/const/subnet120/mining
LOGDIR=$ROOT/experiments/fleet-rent/logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/host_relay_r784_r783_parallel_accel_p3871.log
: >"$LOG"
log() { echo "[p3871-accel] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=45 -o BatchMode=yes
  -o ServerAliveInterval=15 -o ServerAliveCountMax=40 -o TCPKeepAlive=yes)
BRAVE_HOST=18.118.83.97
BRAVE_PORT=40127
CROWN_HOST=95.133.253.90
CROWN_PORT=40099
NPARA=4

stop_pid() {
  local pid=$1 why=${2:-}
  [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    log "kill pid=$pid ($why)"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null || true
  fi
}

log "START stop sequential tar + parallel accel R784 then R783"

# Host outer relay + ssh pipes
while read -r line; do
  pid=${line%% *}
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  echo "$line" | grep -qE 'host_relay_r784_r783_brave_to_crown_p3870\.sh|tar cf - r78[34]_merged' || continue
  stop_pid "$pid" "host:$line"
done < <(pgrep -af 'host_relay_r784_r783_brave_to_crown_p3870|tar cf - r78[34]_merged' || true)

for pid in 3725125 3726318 3726319; do
  stop_pid "$pid" "known-outer-or-ssh"
done

# Crown: kill tar xf only; keep waiters; drop incomplete shards
ssh "${SSH_OPTS[@]}" -p "$CROWN_PORT" "root@$CROWN_HOST" 'bash -s' <<'REMOTE'
set -euo pipefail
log(){ echo "[p3871-crown] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
stop_pid() {
  local pid=$1
  [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    log "kill $pid"; kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 15); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null || true
  fi
}
while read -r pid cmd; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  echo "$cmd" | grep -qE 'wait_r78[34]_scp_then_chall' && continue
  echo "$cmd" | grep -qE 'tar xf' || continue
  stop_pid "$pid"
done < <(ps -eo pid=,args=)
# Also kill the bash -c wrapper that owns tar xf
while read -r pid cmd; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  echo "$cmd" | grep -qE 'cd /tmp && tar xf' || continue
  stop_pid "$pid"
done < <(ps -eo pid=,args=)
src05=4506431736
for f in /tmp/r784_merged/model-*-of-*.safetensors; do
  [[ -f "$f" ]] || continue
  bn=$(basename "$f"); sz=$(stat -c%s "$f")
  if [[ "$bn" == "model-00005-of-00016.safetensors" && "$sz" -eq "$src05" ]]; then
    log "KEEP $bn ($sz)"
  else
    log "RM incomplete $bn ($sz)"; rm -f "$f"
  fi
done
rm -f /tmp/r784_merged/*.tmp /tmp/r783_merged/*.tmp 2>/dev/null || true
mkdir -p /tmp/r784_merged /tmp/r783_merged
rm -f /root/logs/r784_scp_ready.done /root/logs/r783_scp_ready.done
df -h / | tail -1
ls -la /tmp/r784_merged/ | head -20
REMOTE

ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" 'bash -s' <<'REMOTE'
set -euo pipefail
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  kill "$pid" 2>/dev/null || true
  sleep 1
  kill -9 "$pid" 2>/dev/null || true
  echo "killed brave tar pid=$pid"
done < <(ps -eo pid=,args= | awk '/tar cf - r784_merged|tar cf - r783_merged/ && !/awk/ {print $1}')
REMOTE

sleep 2

xfer_one() {
  local hypo=$1 f=$2 want=$3
  local dest="/tmp/${hypo}_merged"
  local attempt rc got
  for attempt in 1 2 3 4 5; do
    log "PIPE $hypo $f attempt=$attempt want=$want"
    set +e
    ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" \
      "cat /tmp/${hypo}_merged/$f" \
      | ssh "${SSH_OPTS[@]}" -p "$CROWN_PORT" "root@$CROWN_HOST" \
        "mkdir -p $dest && cat > $dest/$f.tmp && mv -f $dest/$f.tmp $dest/$f"
    rc=$?
    set -e
    if [[ "$rc" -ne 0 ]]; then
      log "PIPE fail $hypo $f rc=$rc"
      sleep $((attempt * 3))
      continue
    fi
    got=$(ssh "${SSH_OPTS[@]}" -p "$CROWN_PORT" "root@$CROWN_HOST" \
      "stat -c%s $dest/$f 2>/dev/null || echo 0" 2>/dev/null || echo 0)
    got=${got//[^0-9]/}; got=${got:-0}
    if [[ "$got" == "$want" ]]; then
      log "PIPE ok $hypo $f ($got)"
      return 0
    fi
    log "PIPE size mismatch $hypo $f got=$got want=$want"
    sleep $((attempt * 3))
  done
  log "FATAL $hypo $f"
  return 1
}

relay_hypo() {
  local hypo=$1
  log "=== relay $hypo parallel×$NPARA ==="
  mapfile -t SRC_LINES < <(ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" \
    "cd /tmp/${hypo}_merged && for f in model-*-of-*.safetensors model-visual-restored.safetensors config.json tokenizer.json tokenizer_config.json generation_config.json merge_meta.json model.safetensors.index.json chat_template.jinja preprocessor_config.json processor_config.json video_preprocessor_config.json README.md; do
       [[ -f \$f ]] && printf '%s %s\n' \"\$f\" \"\$(stat -c%s \"\$f\")\"
     done")
  log "$hypo src_files=${#SRC_LINES[@]}"
  [[ "${#SRC_LINES[@]}" -ge 17 ]] || { log "FATAL $hypo source incomplete count=${#SRC_LINES[@]}"; return 1; }

  need_list=()
  for line in "${SRC_LINES[@]}"; do
    f=${line%% *}; want=${line##* }
    have=$(ssh "${SSH_OPTS[@]}" -p "$CROWN_PORT" "root@$CROWN_HOST" \
      "stat -c%s /tmp/${hypo}_merged/$f 2>/dev/null || echo 0" 2>/dev/null || echo 0)
    have=${have//[^0-9]/}; have=${have:-0}
    if [[ "$have" == "$want" ]]; then
      log "KEEP $hypo $f ($have)"
      continue
    fi
    if ssh "${SSH_OPTS[@]}" -p "$CROWN_PORT" "root@$CROWN_HOST" \
        "test -f /tmp/${hypo}_merged/$f.tmp && echo yes || echo no" 2>/dev/null | grep -q yes; then
      log "BUSY skip $hypo $f"
      continue
    fi
    log "NEED $hypo $f (have=$have want=$want)"
    need_list+=("$f:$want")
  done
  log "$hypo need_count=${#need_list[@]}"

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
    xfer_one "$hypo" "$f" "$want" &
    pids+=("$!"); active=$((active + 1))
  done
  for p in "${pids[@]:-}"; do
    if ! wait "$p"; then fail=1; fi
  done

  # Push size manifest and verify+stamp on crown
  local manif=/tmp/p3871_${hypo}_sizes.txt
  printf '%s\n' "${SRC_LINES[@]}" >"$manif"
  scp "${SSH_OPTS[@]}" -P "$CROWN_PORT" "$manif" "root@$CROWN_HOST:/tmp/p3871_${hypo}_sizes.txt"
  stamp_out=$(ssh "${SSH_OPTS[@]}" -p "$CROWN_PORT" "root@$CROWN_HOST" \
    "hypo=$hypo bash -s" <<'REMOTE'
set -euo pipefail
ok=1
while read -r f want; do
  [[ -z "${f:-}" ]] && continue
  got=$(stat -c%s /tmp/${hypo}_merged/$f 2>/dev/null || echo 0)
  if [[ "$got" != "$want" ]]; then
    echo "bad:$f:got=$got:want=$want"
    ok=0
  fi
done < /tmp/p3871_${hypo}_sizes.txt
n=$(ls /tmp/${hypo}_merged/model-*-of-*.safetensors 2>/dev/null | wc -l)
vis=0; [[ -f /tmp/${hypo}_merged/model-visual-restored.safetensors ]] && vis=1
cfg=0; [[ -f /tmp/${hypo}_merged/config.json ]] && cfg=1
if [[ "$ok" -eq 1 && "$n" -ge 16 && "$vis" -eq 1 && "$cfg" -eq 1 ]]; then
  # hypo already includes leading r (r784/r783) — do NOT prefix another r
  date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/${hypo}_scp_ready.done
  echo "ok:$n:vis=$vis:$(du -sh /tmp/${hypo}_merged | awk '{print $1}')"
else
  echo "partial:$n:vis=$vis:cfg=$cfg:ok=$ok"
fi
REMOTE
)
  log "stamp_check $hypo $stamp_out"
  [[ "$stamp_out" == ok:* ]] || { log "FATAL $hypo verify fail=$fail stamp=$stamp_out"; return 1; }
  log "DONE $hypo SCP_READY"
  return 0
}

relay_hypo r784
relay_hypo r783

log "DONE dual parallel accel — waiters should arm lean challs"
date -u +%Y-%m-%dT%H:%M:%SZ >"$LOGDIR/host_relay_r784_r783_parallel_accel_p3871.done"
exit 0
