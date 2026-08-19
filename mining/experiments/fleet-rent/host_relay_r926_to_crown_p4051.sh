#!/usr/bin/env bash
# p4051: R926 MERGE_DONE on brave-raven-49 → mine-crown-1 brave-comet-f4 parallel×4.
# Never pkill -f. Leave R926 trains 2–7 untouched. Leave crown TK + R929/R936 trains untouched.
set -euo pipefail
ROOT=/home/const/subnet120/mining
LOGDIR=$ROOT/experiments/fleet-rent/logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/host_relay_r926_to_crown_p4051.log
: >"$LOG"
log() { echo "[p4051-r926] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=45 -o BatchMode=yes
  -o ServerAliveInterval=15 -o ServerAliveCountMax=40 -o TCPKeepAlive=yes)
SRC_HOST=93.120.231.186; SRC_PORT=32301
DST_HOST=95.133.252.28; DST_PORT=40298
NPARA=4
hypo=r926

xfer_one() {
  local f=$1 want=$2 attempt rc got
  for attempt in 1 2 3 4 5; do
    log "PIPE $f attempt=$attempt want=$want"
    set +e
    ssh "${SSH_OPTS[@]}" -p "$SRC_PORT" "root@$SRC_HOST" "cat /tmp/${hypo}_merged/$f" \
      | ssh "${SSH_OPTS[@]}" -p "$DST_PORT" "root@$DST_HOST" \
        "mkdir -p /tmp/${hypo}_merged && cat > /tmp/${hypo}_merged/$f.tmp && mv -f /tmp/${hypo}_merged/$f.tmp /tmp/${hypo}_merged/$f"
    rc=$?; set -e
    [[ "$rc" -eq 0 ]] || { sleep $((attempt*3)); continue; }
    got=$(ssh "${SSH_OPTS[@]}" -p "$DST_PORT" "root@$DST_HOST" "stat -c%s /tmp/${hypo}_merged/$f 2>/dev/null || echo 0")
    got=${got//[^0-9]/}; got=${got:-0}
    [[ "$got" == "$want" ]] && { log "PIPE ok $f"; return 0; }
    sleep $((attempt*3))
  done
  log "FATAL $f"; return 1
}

log "START R926 → crown parallel×$NPARA"
mapfile -t SRC_LINES < <(ssh "${SSH_OPTS[@]}" -p "$SRC_PORT" "root@$SRC_HOST" \
  "cd /tmp/${hypo}_merged && for f in model-*-of-*.safetensors model-visual-restored.safetensors config.json tokenizer.json tokenizer_config.json generation_config.json merge_meta.json model.safetensors.index.json chat_template.jinja preprocessor_config.json processor_config.json video_preprocessor_config.json; do
     [[ -f \$f ]] && printf '%s %s\n' \"\$f\" \"\$(stat -c%s \"\$f\")\"
   done")
need_list=()
for line in "${SRC_LINES[@]}"; do
  f=${line%% *}; want=${line##* }
  need_list+=("$f:$want")
done
log "need_count=${#need_list[@]}"
[[ "${#need_list[@]}" -ge 17 ]] || { log "FATAL need_count=${#need_list[@]} (want ≥17)"; exit 1; }

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

# SIZE_OK even if wait -n false-failed (p4049 lesson)
log "SIZE_OK check (fail=$fail)"
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
  want=$(ssh "${SSH_OPTS[@]}" -p "$SRC_PORT" "root@$SRC_HOST" "stat -c%s /tmp/${hypo}_merged/$f")
  have=$(ssh "${SSH_OPTS[@]}" -p "$DST_PORT" "root@$DST_HOST" "stat -c%s /tmp/${hypo}_merged/$f")
  want=${want//[^0-9]/}; have=${have//[^0-9]/}
  [[ "$want" == "$have" ]] || { log "FATAL mismatch $f $have/$want"; exit 1; }
done
tmps=$(ssh "${SSH_OPTS[@]}" -p "$DST_PORT" "root@$DST_HOST" "ls /tmp/${hypo}_merged/*.tmp 2>/dev/null | wc -l")
tmps=${tmps//[^0-9]/}; tmps=${tmps:-99}
[[ "$tmps" -eq 0 ]] || { log "FATAL tmps=$tmps"; exit 1; }
ssh "${SSH_OPTS[@]}" -p "$DST_PORT" "root@$DST_HOST" "test -f /tmp/${hypo}_merged/config.json"

stamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
ssh "${SSH_OPTS[@]}" -p "$DST_PORT" "root@$DST_HOST" \
  "echo '$stamp' > /root/logs/${hypo}_scp_ready.done && echo stamped:\$(cat /root/logs/${hypo}_scp_ready.done):\$(ls /tmp/${hypo}_merged/model-*-of-*.safetensors | wc -l)"
log "STAMPED r926_scp_ready.done=$stamp"
date -u +%Y-%m-%dT%H:%M:%SZ >"$LOGDIR/host_relay_r926_to_crown_p4051.done"
log "DONE R926 → crown relay"
