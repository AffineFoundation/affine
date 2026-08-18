#!/usr/bin/env bash
# p3880: After R783 n80 resolves — if not a clear WIN, free 6,7 and fast×4 relay
# R801 MERGE_DONE brave→crown. Wait for R800 relay done first (no dual-pipe).
# Never pkill -f. Leave R800 n80 on 4,5 alone.
set -euo pipefail
ROOT=/home/const/subnet120/mining
LOGDIR=$ROOT/experiments/fleet-rent/logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/host_relay_r801_after_r783_p3880.log
: >"$LOG"
log() { echo "[p3880-r801] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=45 -o BatchMode=yes
  -o ServerAliveInterval=15 -o ServerAliveCountMax=40 -o TCPKeepAlive=yes)
BRAVE_HOST=18.118.83.97; BRAVE_PORT=40127
CROWN_HOST=95.133.253.90; CROWN_PORT=40099
NPARA=4
hypo=r801
R800_DONE=$LOGDIR/host_relay_r800_fast_p3879.done
R783_RESULT=/root/affine_data/r783_sim_result_reign35_wvk7.json

log "WAIT R783 result on crown"
while true; do
  if ssh "${SSH_OPTS[@]}" -p "$CROWN_PORT" "root@$CROWN_HOST" "test -f $R783_RESULT"; then
    break
  fi
  sleep 20
done
log "R783 result present — score decision"

dec=$(ssh "${SSH_OPTS[@]}" -p "$CROWN_PORT" "root@$CROWN_HOST" python3 - <<'PY'
import json
from pathlib import Path
p = Path("/root/affine_data/r783_sim_result_reign35_wvk7.json")
d = json.loads(p.read_text())
v = d.get("verdict") if isinstance(d.get("verdict"), dict) else {}
chal = (v.get("challenger") or {}) if isinstance(v, dict) else {}
margin = v.get("margin") if v else (d.get("margin") or d.get("mean_margin"))
se = v.get("se") if v else (d.get("se") or d.get("stderr"))
try:
    m = float(margin); s = float(se)
    bar = max(2.0 * s, 0.002)
except Exception:
    m = s = bar = None
thought = chal.get("median_len_z")
bpass = chal.get("b_gate_pass_rate")
win = False
try:
    win = (m is not None and bar is not None and m > bar
           and thought is not None and float(thought) >= 80
           and bpass is not None and float(bpass) >= 0.30)
except Exception:
    win = False
print(json.dumps({"margin": m, "se": s, "bar": bar, "thought": thought, "b_pass": bpass, "win": win}))
PY
)
log "R783 decision=$dec"
if echo "$dec" | python3 -c "import sys,json; d=json.load(sys.stdin); raise SystemExit(0 if d.get('win') else 1)"; then
  log "R783 CLEAR WIN — leave 6,7 for Stage-5; abort R801 relay"
  date -u +%Y-%m-%dT%H:%M:%SZ >"$LOGDIR/host_relay_r801_after_r783_p3880.aborted_win"
  exit 0
fi

log "R783 not crown-clear — free chall on 6,7 by exact PID"
ssh "${SSH_OPTS[@]}" -p "$CROWN_PORT" "root@$CROWN_HOST" bash -s <<'REMOTE'
set -euo pipefail
log(){ echo "[p3880-free-r783] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
stop_pid() {
  local pid=$1
  [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    log "kill $pid"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null || true
  fi
}
for pf in /root/logs/vllm_chall_r783.pid /root/logs/r783_sim_wvk7.pid; do
  [[ -f "$pf" ]] || continue
  stop_pid "$(cat "$pf" 2>/dev/null || true)"
  rm -f "$pf"
done
# Exact argv kills only (never pkill -f)
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid"
done < <(ps -eo pid=,args= | awk '/\/tmp\/r783_merged/ && !/awk/ {print $1}')
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid"
done < <(ps -eo pid=,args= | awk '/run_sim_duel.py.*r783/ && !/awk/ {print $1}')
for i in $(seq 1 60); do
  used=$(nvidia-smi -i 6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  if [[ "${used:-999999}" -lt 2000 ]]; then log "GPUs 6,7 free used=$used"; break; fi
  sleep 2
done
REMOTE

log "WAIT R800 relay done ($R800_DONE) before R801 pipes"
while [[ ! -f "$R800_DONE" ]]; do sleep 20; done
log "R800 relay done — START R801 fast×$NPARA"

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
log "STAMPED r801_scp_ready.done=$stamp"
date -u +%Y-%m-%dT%H:%M:%SZ >"$LOGDIR/host_relay_r801_after_r783_p3880.done"
log "DONE R801 deferred relay"
