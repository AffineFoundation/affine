#!/usr/bin/env bash
# p3878: after R783 SCP_READY on crown, host-relay R800 MERGE_DONE brave→crown parallel×4,
# size-verify 16+vis, stamp r800_scp_ready.done. Never pkill -f. Leave R783 n80 / live pipes alone.
set -euo pipefail
ROOT=/home/const/subnet120/mining
LOGDIR=$ROOT/experiments/fleet-rent/logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/host_relay_r800_after_r783_p3878.log
: >"$LOG"
log() { echo "[p3878-r800] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=45 -o BatchMode=yes
  -o ServerAliveInterval=15 -o ServerAliveCountMax=40 -o TCPKeepAlive=yes)
BRAVE_HOST=18.118.83.97; BRAVE_PORT=40127
CROWN_HOST=95.133.253.90; CROWN_PORT=40099
NPARA=4
hypo=r800

log "WAIT for crown /root/logs/r783_scp_ready.done (avoid dual-pipe with R783)"
while true; do
  if ssh "${SSH_OPTS[@]}" -p "$CROWN_PORT" "root@$CROWN_HOST" \
      "test -f /root/logs/r783_scp_ready.done && echo yes || echo no" 2>/dev/null | grep -q yes; then
    stamp=$(ssh "${SSH_OPTS[@]}" -p "$CROWN_PORT" "root@$CROWN_HOST" "cat /root/logs/r783_scp_ready.done" 2>/dev/null || true)
    log "R783 stamped: $stamp — begin R800 relay"
    break
  fi
  # also wait until no r783 .tmp (pipes fully drained)
  sleep 20
done

# Drain any lingering r783 .tmp before starting (bandwidth)
for i in $(seq 1 90); do
  tmps=$(ssh "${SSH_OPTS[@]}" -p "$CROWN_PORT" "root@$CROWN_HOST" \
    "ls /tmp/r783_merged/*.tmp 2>/dev/null | wc -l" 2>/dev/null || echo 99)
  tmps=${tmps//[^0-9]/}; tmps=${tmps:-99}
  [[ "$tmps" -eq 0 ]] && { log "r783 tmps=0 — uplink free"; break; }
  log "wait r783 tmps=$tmps iter=$i"
  sleep 10
done

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

log "START R800 parallel×$NPARA"
mapfile -t SRC_LINES < <(ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" \
  "cd /tmp/${hypo}_merged && for f in model-*-of-*.safetensors model-visual-restored.safetensors config.json tokenizer.json tokenizer_config.json generation_config.json merge_meta.json model.safetensors.index.json chat_template.jinja preprocessor_config.json processor_config.json video_preprocessor_config.json README.md; do
     [[ -f \$f ]] && printf '%s %s\n' \"\$f\" \"\$(stat -c%s \"\$f\")\"
   done")
need_list=()
for line in "${SRC_LINES[@]}"; do
  f=${line%% *}; want=${line##* }
  have=$(ssh "${SSH_OPTS[@]}" -p "$CROWN_PORT" "root@$CROWN_HOST" \
    "stat -c%s /tmp/${hypo}_merged/$f 2>/dev/null || echo 0" 2>/dev/null || echo 0)
  have=${have//[^0-9]/}; have=${have:-0}
  [[ "$have" == "$want" ]] && { log "KEEP $f"; continue; }
  if ssh "${SSH_OPTS[@]}" -p "$CROWN_PORT" "root@$CROWN_HOST" \
      "test -f /tmp/${hypo}_merged/$f.tmp && echo yes || echo no" 2>/dev/null | grep -q yes; then
    log "BUSY skip $f"; continue
  fi
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

# Size-verify all 16+vis vs brave
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

# Ensure config present
ssh "${SSH_OPTS[@]}" -p "$CROWN_PORT" "root@$CROWN_HOST" "test -f /tmp/${hypo}_merged/config.json"

stamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
ssh "${SSH_OPTS[@]}" -p "$CROWN_PORT" "root@$CROWN_HOST" \
  "echo '$stamp' > /root/logs/${hypo}_scp_ready.done && echo stamped:\$(cat /root/logs/${hypo}_scp_ready.done):\$(ls /tmp/${hypo}_merged/model-*-of-*.safetensors | wc -l)"
log "STAMPED r800_scp_ready.done=$stamp"
date -u +%Y-%m-%dT%H:%M:%SZ >"$LOGDIR/host_relay_r800_after_r783_p3878.done"
log "DONE R800 relay"
