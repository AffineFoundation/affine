#!/usr/bin/env bash
# p3865: pipe small R781 meta files brave→R252 while weight parallel×4 (p3848) + tail (p3865) continue.
# SSH-only (R252 lium exec flaky). Never touch model-*-of-*.safetensors. Never pkill -f.
set -euo pipefail
ROOT=/home/const/subnet120/mining
LOGDIR=$ROOT/experiments/fleet-rent/logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/host_relay_r781_meta_accel_p3865.log
: >"$LOG"
log() { echo "[p3865-r781-meta] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=45 -o BatchMode=yes
  -o ServerAliveInterval=15 -o ServerAliveCountMax=40 -o TCPKeepAlive=yes)
BRAVE_HOST=18.118.83.97
BRAVE_PORT=40127
R252_HOST=95.133.252.28
R252_PORT=40299

# Small files only — do NOT touch model-*-of-*.safetensors or model-visual
META=(
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
  README.md
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

xfer_one() {
  local f="$1" want="$2"
  local attempt rc got
  for attempt in 1 2 3; do
    log "PIPE start $f attempt=$attempt want=$want"
    set +e
    ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" \
      "cat /tmp/r781_merged/$f" \
      | ssh "${SSH_OPTS[@]}" -p "$R252_PORT" "root@$R252_HOST" \
        "mkdir -p /tmp/r781_merged && cat > /tmp/r781_merged/$f.tmp && mv -f /tmp/r781_merged/$f.tmp /tmp/r781_merged/$f"
    rc=$?
    set -e
    if [[ "$rc" -ne 0 ]]; then
      log "PIPE fail $f rc=$rc"
      sleep $((attempt * 2))
      continue
    fi
    got=$(dest_size "$f")
    got=${got:-0}
    if [[ "$got" == "$want" ]]; then
      log "PIPE ok $f ($got)"
      return 0
    fi
    log "PIPE size mismatch $f got=$got want=$want"
    sleep $((attempt * 2))
  done
  log "FATAL $f"
  return 1
}

log "START meta accel (weights left to p3848 + p3865-tail)"
fail=0
pids=()
for f in "${META[@]}"; do
  want=$(src_size "$f")
  want=${want:-0}
  if [[ "$want" -eq 0 ]]; then
    log "SKIP missing src $f"
    continue
  fi
  have=$(dest_size "$f")
  have=${have:-0}
  if [[ "$have" == "$want" ]]; then
    log "KEEP $f ($have)"
    continue
  fi
  xfer_one "$f" "$want" &
  pids+=("$!")
done
for p in "${pids[@]:-}"; do
  if ! wait "$p"; then fail=1; fi
done
log "DONE meta accel fail=$fail"
exit 0
