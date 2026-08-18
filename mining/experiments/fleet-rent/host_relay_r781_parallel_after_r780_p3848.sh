#!/usr/bin/env bash
# p3848: R781 MERGE_DONE (66G/16 shards/vis) sat idle on brave with no n80 waiter.
# Wait R780 host DONE + SCP_READY, then parallel×4 size-checked pipes brave→R252.
# Never pkill -f. Leave R767/R768/R780/R793 alone. Queue after R780.
set -euo pipefail
ROOT=/home/const/subnet120/mining
EXP=$ROOT/experiments/r781-tammy-offline-dpo-hialpha-hirank-hibeta-softctx-megasuperextrasteps-ep4-lolr
LOGDIR=$ROOT/experiments/fleet-rent/logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/host_relay_r781_parallel_after_r780_p3848.log
R780_LOG=$LOGDIR/host_relay_r780_parallel_after_r768_p3848.log
: >"$LOG"
log() { echo "[p3848-r781] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=45 -o BatchMode=yes
  -o ServerAliveInterval=15 -o ServerAliveCountMax=40 -o TCPKeepAlive=yes)
BRAVE_HOST=18.118.83.97
BRAVE_PORT=40127
R252_HOST=95.133.252.28
R252_PORT=40299
HUID=gentle-wolf-8c
NPARA=4

r780_host_done() {
  if grep -qE 'DONE relay|DONE parallel|SCP_READY' "$R780_LOG" 2>/dev/null; then return 0; fi
  if grep -qE 'FATAL' "$R780_LOG" 2>/dev/null && ! grep -qE 'DONE|SCP_READY' "$R780_LOG" 2>/dev/null; then
    # only fatal if no success later
    if grep -qE 'FATAL verify|FATAL some transfers|FATAL source' "$R780_LOG" 2>/dev/null; then return 2; fi
  fi
  return 1
}

r252_r780_ready() {
  local out=""
  out=$(timeout 55 lium exec "$HUID" 'if [[ -f /root/logs/r780_scp_ready.done ]] && [[ -f /tmp/r780_merged/config.json ]]; then
    n=$(ls /tmp/r780_merged/model-*-of-*.safetensors 2>/dev/null | wc -l); echo "ok:$n"
  else
    n=$(ls /tmp/r780_merged/model-*-of-*.safetensors 2>/dev/null | wc -l || echo 0); echo "wait:$n"
  fi' 2>/dev/null | grep -E '^(ok|wait):' | tail -1 || true)
  if [[ -n "$out" ]]; then echo "$out"; return 0; fi
  out=$(ssh "${SSH_OPTS[@]}" -p "$R252_PORT" "root@$R252_HOST" \
    'if [[ -f /root/logs/r780_scp_ready.done ]] && [[ -f /tmp/r780_merged/config.json ]]; then
       n=$(ls /tmp/r780_merged/model-*-of-*.safetensors 2>/dev/null | wc -l); echo "ok:$n"
     else
       n=$(ls /tmp/r780_merged/model-*-of-*.safetensors 2>/dev/null | wc -l || echo 0); echo "wait:$n"
     fi' 2>/dev/null || true)
  if [[ -n "$out" ]]; then echo "$out"; return 0; fi
  echo "sshfail"
}

log "armed: wait R780 SCP_READY + host DONE then parallel×4 R781 (timeout 10h)"
ready=""
for i in $(seq 1 2400); do
  hd=0
  set +e; r780_host_done; hd_rc=$?; set -e
  if [[ "$hd_rc" -eq 2 ]]; then log "FATAL R780 host relay failed"; exit 1
  elif [[ "$hd_rc" -eq 0 ]]; then hd=1; fi
  ready=$(r252_r780_ready)
  if [[ "$ready" == ok:* ]]; then
    shards=${ready#ok:}
    if [[ "${shards:-0}" -ge 16 && "$hd" -eq 1 ]]; then
      log "R780 ready shards=$shards host_done=1 — begin R781 (poll=$i)"
      break
    fi
  fi
  [[ $((i % 8)) -eq 0 ]] && log "poll i=$i r780=$ready host_done=$hd"
  if [[ "$i" -eq 2400 ]]; then log "TIMEOUT waiting R780 last=$ready hd=$hd"; exit 1; fi
  sleep 15
done

# Wait prior relays gone so we do not fight SSH uplink
for i in $(seq 1 240); do
  if ! pgrep -f 'host_relay_r780_parallel_after_r768_p3848\.sh' >/dev/null 2>&1 \
     && ! pgrep -f 'host_relay_r767_parallel_resume_p3847\.sh' >/dev/null 2>&1 \
     && ! pgrep -f 'host_relay_r768_resume_after_r767_p3837\.sh' >/dev/null 2>&1; then
    log "prior host relays idle — uplink free"
    break
  fi
  [[ $((i % 6)) -eq 0 ]] && log "wait prior relays exit poll=$i"
  sleep 10
done

n_src=$(ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" \
  "ls /tmp/r781_merged/model-*-of-*.safetensors 2>/dev/null | wc -l")
log "brave r781 shards=$n_src"
[[ "$n_src" -ge 16 ]] || { log "FATAL source incomplete"; exit 1; }

stage_ok=0
for attempt in $(seq 1 30); do
  if ssh "${SSH_OPTS[@]}" -p "$R252_PORT" "root@$R252_HOST" \
      "mkdir -p /root/mining_src/r781-chall /root/logs /root/affine_data" \
    && scp "${SSH_OPTS[@]}" -P "$R252_PORT" \
      "$EXP/lean_chall_n80_r252_gpus45_p3848.sh" \
      "$EXP/wait_r781_after_r780_then_n80_p3848.sh" \
      "root@$R252_HOST:/root/mining_src/r781-chall/" \
    && ssh "${SSH_OPTS[@]}" -p "$R252_PORT" "root@$R252_HOST" 'bash -s' <<'REMOTE'
set -euo pipefail
chmod +x /root/mining_src/r781-chall/*.sh
bash -n /root/mining_src/r781-chall/lean_chall_n80_r252_gpus45_p3848.sh
bash -n /root/mining_src/r781-chall/wait_r781_after_r780_then_n80_p3848.sh
echo SYNTAX_OK
rm -rf /tmp/r781_merged
rm -f /root/logs/r781_scp_ready.done /root/logs/r781_chall_n80_launched.p3848
mkdir -p /tmp/r781_merged
df -h / | tail -1
nohup bash /root/mining_src/r781-chall/wait_r781_after_r780_then_n80_p3848.sh \
  >/root/logs/p3848_r781_wait.nohup 2>&1 &
echo $! >/root/logs/p3848_r781_wait.pid
echo "waiter_started pid=$(cat /root/logs/p3848_r781_wait.pid)"
REMOTE
  then stage_ok=1; log "stage scripts ok attempt=$attempt"; break
  fi
  log "stage scripts fail attempt=$attempt — sleep 20"; sleep 20
done
[[ "$stage_ok" -eq 1 ]] || { log "FATAL could not stage R781 scripts on R252"; exit 3; }

mapfile -t SRC_LINES < <(ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" \
  'cd /tmp/r781_merged && for f in model-*-of-*.safetensors model-visual-restored.safetensors config.json tokenizer.json tokenizer_config.json generation_config.json merge_meta.json model.safetensors.index.json chat_template.jinja preprocessor_config.json processor_config.json video_preprocessor_config.json README.md; do
     [[ -f "$f" ]] && printf "%s %s\n" "$f" "$(stat -c%s "$f")"
   done')
log "brave files=${#SRC_LINES[@]}"
[[ ${#SRC_LINES[@]} -ge 17 ]] || { log "FATAL source map short n=${#SRC_LINES[@]}"; exit 1; }

dest_size() {
  local f="$1" out
  out=$(timeout 55 lium exec "$HUID" "stat -c%s /tmp/r781_merged/$f 2>/dev/null || echo 0" 2>/dev/null \
    | grep -E '^[0-9]+$' | tail -1 || true)
  if [[ -n "$out" ]]; then echo "$out"; return 0; fi
  out=$(ssh "${SSH_OPTS[@]}" -p "$R252_PORT" "root@$R252_HOST" \
    "stat -c%s /tmp/r781_merged/$f 2>/dev/null || echo 0" 2>/dev/null || echo 0)
  echo "${out//[^0-9]/}"
}

rm_dest() {
  local f="$1"
  timeout 55 lium exec "$HUID" "rm -f /tmp/r781_merged/$f" >/dev/null 2>&1 || true
  ssh "${SSH_OPTS[@]}" -p "$R252_PORT" "root@$R252_HOST" "rm -f /tmp/r781_merged/$f" 2>/dev/null || true
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
      "cat /tmp/r781_merged/$f" \
      | ssh "${SSH_OPTS[@]}" -p "$R252_PORT" "root@$R252_HOST" \
        "mkdir -p /tmp/r781_merged && cat > /tmp/r781_merged/$f.tmp && mv -f /tmp/r781_merged/$f.tmp /tmp/r781_merged/$f"
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
  out=$(timeout 90 lium exec "$HUID" 'n=$(ls /tmp/r781_merged/model-*-of-*.safetensors 2>/dev/null | wc -l)
    vis=0; [[ -f /tmp/r781_merged/model-visual-restored.safetensors ]] && vis=1
    if [[ -f /tmp/r781_merged/config.json ]] && [[ "$n" -ge 16 ]]; then
      date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r781_scp_ready.done
      echo "ok:$n:vis=$vis:$(du -sh /tmp/r781_merged | awk "{print \$1}")"
    else
      echo "bad:$n:vis=$vis"
    fi' 2>/dev/null | grep -E '^(ok|bad):' | tail -1 || true)
  if [[ "$out" == ok:* ]]; then
    echo "$out"
    return 0
  fi
  out=$(ssh "${SSH_OPTS[@]}" -p "$R252_PORT" "root@$R252_HOST" \
    'n=$(ls /tmp/r781_merged/model-*-of-*.safetensors 2>/dev/null | wc -l)
     vis=0; [[ -f /tmp/r781_merged/model-visual-restored.safetensors ]] && vis=1
     if [[ -f /tmp/r781_merged/config.json ]] && [[ "$n" -ge 16 ]]; then
       date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r781_scp_ready.done
       echo "ok:$n:vis=$vis:$(du -sh /tmp/r781_merged | awk "{print \$1}")"
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
    log "DONE parallel resume — waiter may launch R781 n80 after R780"
    exit 0
  fi
  log "verify attempt=$i got=$ready"
  sleep 5
done
log "FATAL verify failed"
exit 5
