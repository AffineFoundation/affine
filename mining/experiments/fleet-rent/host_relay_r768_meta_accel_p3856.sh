#!/usr/bin/env bash
# p3856: pipe small R768 meta files brave→R252 while weight parallel×4 continues.
# Unblocks config.json early for stamp once shards land. Never pkill -f.
set -euo pipefail
ROOT=/home/const/subnet120/mining
LOGDIR=$ROOT/experiments/fleet-rent/logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/host_relay_r768_meta_accel_p3856.log
: >"$LOG"
log() { echo "[p3856-r768-meta] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=45 -o BatchMode=yes
  -o ServerAliveInterval=15 -o ServerAliveCountMax=40 -o TCPKeepAlive=yes)
BRAVE_HOST=18.118.83.97
BRAVE_PORT=40127
R252_HOST=95.133.252.28
R252_PORT=40299

# Small files only — do NOT touch model-*-of-*.safetensors (p3854 owns those)
META=(
  config.json
  tokenizer_config.json
  generation_config.json
  merge_meta.json
  model.safetensors.index.json
  chat_template.jinja
  preprocessor_config.json
  processor_config.json
  video_preprocessor_config.json
)

dest_size() {
  local f="$1" out
  out=$(ssh "${SSH_OPTS[@]}" -p "$R252_PORT" "root@$R252_HOST" \
    "stat -c%s /tmp/r768_merged/$f 2>/dev/null || echo 0" 2>/dev/null || echo 0)
  echo "${out//[^0-9]/}"
}

src_size() {
  local f="$1" out
  out=$(ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" \
    "stat -c%s /tmp/r768_merged/$f 2>/dev/null || echo 0" 2>/dev/null || echo 0)
  echo "${out//[^0-9]/}"
}

xfer_one() {
  local f="$1" want="$2"
  local attempt rc got
  for attempt in 1 2 3; do
    log "PIPE start $f attempt=$attempt want=$want"
    set +e
    ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" \
      "cat /tmp/r768_merged/$f" \
      | ssh "${SSH_OPTS[@]}" -p "$R252_PORT" "root@$R252_HOST" \
        "mkdir -p /tmp/r768_merged && cat > /tmp/r768_merged/$f.tmp && mv -f /tmp/r768_merged/$f.tmp /tmp/r768_merged/$f"
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

log "START meta accel (weight pipes left to p3854)"
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
# Also pull visual restored (medium) if free — helps lean serve after stamp
vis_want=$(src_size model-visual-restored.safetensors)
vis_want=${vis_want:-0}
if [[ "$vis_want" -gt 0 ]]; then
  have=$(dest_size model-visual-restored.safetensors)
  have=${have:-0}
  if [[ "$have" != "$vis_want" ]]; then
    log "PIPE visual-restored (medium) want=$vis_want"
    xfer_one model-visual-restored.safetensors "$vis_want" || fail=1
  else
    log "KEEP model-visual-restored.safetensors ($have)"
  fi
fi
# Status snapshot for next pass
ssh "${SSH_OPTS[@]}" -p "$R252_PORT" "root@$R252_HOST" \
  'n=$(ls /tmp/r768_merged/model-*-of-*.safetensors 2>/dev/null | wc -l)
   cfg=0; [[ -f /tmp/r768_merged/config.json ]] && cfg=1
   vis=0; [[ -f /tmp/r768_merged/model-visual-restored.safetensors ]] && vis=1
   echo "dest shards=$n cfg=$cfg vis=$vis du=$(du -sh /tmp/r768_merged | awk "{print \$1}")"' \
  | tee -a "$LOG" || true
[[ "$fail" -eq 0 ]] || { log "FATAL some meta failed"; exit 4; }
log "DONE meta accel"
