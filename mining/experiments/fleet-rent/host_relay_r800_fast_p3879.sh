#!/usr/bin/env bash
# p3879: R800 MERGE brave→crown parallel×4 (skip slow per-file inventory; crown empty).
# After SIZE_OK stamp r800_scp_ready.done. Never pkill -f. Leave R783 n80 on 6,7 alone.
set -euo pipefail
ROOT=/home/const/subnet120/mining
LOGDIR=$ROOT/experiments/fleet-rent/logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/host_relay_r800_fast_p3879.log
: >"$LOG"
log() { echo "[p3879-r800] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=45 -o BatchMode=yes
  -o ServerAliveInterval=15 -o ServerAliveCountMax=40 -o TCPKeepAlive=yes)
BRAVE_HOST=18.118.83.97; BRAVE_PORT=40127
CROWN_HOST=95.133.253.90; CROWN_PORT=40099
NPARA=4
hypo=r800

xfer_one() {
  local f=$1 want=$2 attempt rc got
  for attempt in 1 2 3 4 5; do
    log "PIPE $f attempt=$attempt want=$want"
    set +e
    ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" "cat /tmp/${hypo}_merged/$f" \
      | ssh "${SSH_OPTS[@]}" -p "$CROWN_PORT" "root@$CROWN_HOST" \
        "mkdir -p /tmp/${hypo}_merged && cat > /tmp/${hypo}_merged/$f.tmp && mv -f /tmp/${hypo}_merged/$f.tmp /tmp/${hypo}_merged/$f"
    rc=$?; set -e
    [[ "$rc" -eq 0 ]] || { sleep $((attempt*3)); continue; }
    got=$(ssh "${SSH_OPTS[@]}" -p "$CROWN_PORT" "root@$CROWN_HOST" "stat -c%s /tmp/${hypo}_merged/$f 2>/dev/null || echo 0")
    got=${got//[^0-9]/}; got=${got:-0}
    [[ "$got" == "$want" ]] && { log "PIPE ok $f"; return 0; }
    sleep $((attempt*3))
  done
  log "FATAL $f"; return 1
}

log "START R800 fast parallel×$NPARA (no per-file crown inventory)"
mapfile -t SRC_LINES < <(ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" \
  "cd /tmp/${hypo}_merged && for f in model-*-of-*.safetensors model-visual-restored.safetensors config.json tokenizer.json tokenizer_config.json generation_config.json merge_meta.json model.safetensors.index.json chat_template.jinja preprocessor_config.json processor_config.json video_preprocessor_config.json; do
     [[ -f \$f ]] && printf '%s %s\n' \"\$f\" \"\$(stat -c%s \"\$f\")\"
   done")
need_list=()
for line in "${SRC_LINES[@]}"; do
  f=${line%% *}; want=${line##* }
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
      if kill -0 "$p" 2>/dev/null; then live+=("$p"); active=$((active+1)); fi
    done
    pids=("${live[@]:-}")
  done
  xfer_one "$f" "$want" &
  pids+=("$!"); active=$((active+1))
done
for p in "${pids[@]:-}"; do
  if ! wait "$p"; then fail=1; fi
done
[[ "$fail" -eq 0 ]] || { log "FATAL pipes fail=$fail"; exit 1; }

log "SIZE_OK check"
FILES=(
  model-00001-of-00016.safetensors model-00002-of-00016.safetensors
  model-00003-of-00016.safetensors model-00004-of-00016.safetensors
  model-00005-of-00016.safetensors model-00006-of-00016.safetensors
  model-00007-of-00016.safetensors model-00008-of-00016.safetensors
  model-00009-of-00016.safetensors model-00010-of-00016.safetensors
  model-00011-of-00016.safetensors model-00012-of-00016.safetensors
  model-00013-of-00016.safetensors model-00014-of-00016.safetensors
  model-00015-of-00016.safetensors model-00016-of-00016.safetensors
  model-visual-restored.safetensors
)
for f in "${FILES[@]}"; do
  want=$(ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" "stat -c%s /tmp/${hypo}_merged/$f")
  have=$(ssh "${SSH_OPTS[@]}" -p "$CROWN_PORT" "root@$CROWN_HOST" "stat -c%s /tmp/${hypo}_merged/$f")
  want=${want//[^0-9]/}; have=${have//[^0-9]/}
  [[ "$want" == "$have" ]] || { log "FATAL mismatch $f $have/$want"; exit 1; }
done
tmps=$(ssh "${SSH_OPTS[@]}" -p "$CROWN_PORT" "root@$CROWN_HOST" "ls /tmp/${hypo}_merged/*.tmp 2>/dev/null | wc -l")
tmps=${tmps//[^0-9]/}; tmps=${tmps:-99}
[[ "$tmps" -eq 0 ]] || { log "FATAL tmps=$tmps"; exit 1; }
ssh "${SSH_OPTS[@]}" -p "$CROWN_PORT" "root@$CROWN_HOST" "test -f /tmp/${hypo}_merged/config.json"

stamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
ssh "${SSH_OPTS[@]}" -p "$CROWN_PORT" "root@$CROWN_HOST" \
  "echo '$stamp' > /root/logs/${hypo}_scp_ready.done && echo stamped:\$(cat /root/logs/${hypo}_scp_ready.done):\$(ls /tmp/${hypo}_merged/model-*-of-*.safetensors | wc -l)"
log "STAMPED r800_scp_ready.done=$stamp"
date -u +%Y-%m-%dT%H:%M:%SZ >"$LOGDIR/host_relay_r800_fast_p3879.done"
log "DONE R800 fast relay"
