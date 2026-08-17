#!/usr/bin/env bash
# p3696: R653 dest mid-SCP missing model-00004 while later shards already present
# (same hole class as R651/R634). Wait for live brave→golden tar to exit (no
# dual-pipe), verify ALL 16 shards against source byte sizes, patch only
# missing/truncated files, then write r653_scp_ready.done.
# Never pkill -f. Detect tar via `pgrep -c -f '^tar xf'` (do NOT put
# r653_merged in that remote argv — self-match stuck R634 repair).
set -euo pipefail

ROOT=/home/const/subnet120/mining
LOGDIR=$ROOT/experiments/fleet-rent/logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/repair_r653_missing_shards_p3696.log
: >"$LOG"
log() { echo "[p3696-r653-repair] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=45 -o BatchMode=yes
  -o ServerAliveInterval=15 -o ServerAliveCountMax=40 -o TCPKeepAlive=yes)
BRAVE_HOST=18.118.83.97
BRAVE_PORT=40127
GOLD_HOST=38.127.229.127
GOLD_PORT=40299
SRC=/tmp/r653_merged
DST=/tmp/r653_merged
SCP_DONE=/root/logs/r653_scp_ready.done
LAUNCHED=/root/logs/r653_chall_n80_launched.p3693

# Source sizes from brave 2026-08-17 (immutable merge).
declare -A EXP=(
  [model-00001-of-00016.safetensors]=4323955448
  [model-00002-of-00016.safetensors]=4506431768
  [model-00003-of-00016.safetensors]=4988775056
  [model-00004-of-00016.safetensors]=3962207584
  [model-00005-of-00016.safetensors]=4506431736
  [model-00006-of-00016.safetensors]=4988775104
  [model-00007-of-00016.safetensors]=3962207632
  [model-00008-of-00016.safetensors]=4506431816
  [model-00009-of-00016.safetensors]=4988775104
  [model-00010-of-00016.safetensors]=3962207632
  [model-00011-of-00016.safetensors]=4506431816
  [model-00012-of-00016.safetensors]=4988775104
  [model-00013-of-00016.safetensors]=3962207632
  [model-00014-of-00016.safetensors]=4506431816
  [model-00015-of-00016.safetensors]=4988775104
  [model-00016-of-00016.safetensors]=1672494048
)

log "START — wait for live R653 tar recv to exit, then size-check+patch"

for i in $(seq 1 720); do
  alive=$(ssh "${SSH_OPTS[@]}" -p "$GOLD_PORT" "root@$GOLD_HOST" \
    'pgrep -c -f "^tar xf" 2>/dev/null || true' 2>/dev/null || echo 0)
  brave_tar=$(ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" \
    "ps aux | grep '[t]ar cf - r653_merged' | wc -l" 2>/dev/null || echo 0)
  n=$(ssh "${SSH_OPTS[@]}" -p "$GOLD_PORT" "root@$GOLD_HOST" \
    "ls $DST/model-*-of-*.safetensors 2>/dev/null | wc -l" 2>/dev/null || echo 0)
  sz=$(ssh "${SSH_OPTS[@]}" -p "$GOLD_PORT" "root@$GOLD_HOST" \
    "du -sh $DST 2>/dev/null | awk '{print \$1}'" 2>/dev/null || echo '?')
  has4=$(ssh "${SSH_OPTS[@]}" -p "$GOLD_PORT" "root@$GOLD_HOST" \
    "test -f $DST/model-00004-of-00016.safetensors && echo yes || echo no" 2>/dev/null || echo '?')
  [[ $((i % 3)) -eq 1 ]] && log "poll i=$i tar_xf=$alive brave_tar=$brave_tar shards=$n size=$sz shard4=$has4"
  if [[ "${alive:-0}" -eq 0 && "${brave_tar:-0}" -eq 0 ]]; then
    log "tar send+recv gone — begin completeness check"
    break
  fi
  if [[ "$i" -eq 720 ]]; then
    log "FATAL timeout waiting for tar to exit"
    exit 1
  fi
  sleep 20
done

# List missing / truncated relative to expected sizes
mapfile -t missing < <(ssh "${SSH_OPTS[@]}" -p "$GOLD_PORT" "root@$GOLD_HOST" bash -s <<'REMOTE'
set -euo pipefail
DST=/tmp/r653_merged
declare -A EXP=(
  [model-00001-of-00016.safetensors]=4323955448
  [model-00002-of-00016.safetensors]=4506431768
  [model-00003-of-00016.safetensors]=4988775056
  [model-00004-of-00016.safetensors]=3962207584
  [model-00005-of-00016.safetensors]=4506431736
  [model-00006-of-00016.safetensors]=4988775104
  [model-00007-of-00016.safetensors]=3962207632
  [model-00008-of-00016.safetensors]=4506431816
  [model-00009-of-00016.safetensors]=4988775104
  [model-00010-of-00016.safetensors]=3962207632
  [model-00011-of-00016.safetensors]=4506431816
  [model-00012-of-00016.safetensors]=4988775104
  [model-00013-of-00016.safetensors]=3962207632
  [model-00014-of-00016.safetensors]=4506431816
  [model-00015-of-00016.safetensors]=4988775104
  [model-00016-of-00016.safetensors]=1672494048
)
need_meta=(
  config.json chat_template.jinja processor_config.json video_preprocessor_config.json
  model-visual-restored.safetensors model.safetensors.index.json generation_config.json
  tokenizer.json tokenizer_config.json preprocessor_config.json merge_meta.json
)
for f in "${need_meta[@]}"; do
  if [[ ! -f "$DST/$f" ]]; then echo "$f"; continue; fi
  sz=$(stat -c%s "$DST/$f" 2>/dev/null || echo 0)
  [[ "$sz" -lt 64 ]] && echo "$f"
done
for f in "${!EXP[@]}"; do
  if [[ ! -f "$DST/$f" ]]; then echo "$f"; continue; fi
  sz=$(stat -c%s "$DST/$f" 2>/dev/null || echo 0)
  # accept within 0.5% (tar mid-write should be done by now; still catch trunc)
  want=${EXP[$f]}
  min=$(( want * 995 / 1000 ))
  [[ "$sz" -lt "$min" ]] && echo "$f"
done
REMOTE
)

mapfile -t missing < <(printf '%s\n' "${missing[@]:-}" | awk 'NF && !seen[$0]++')

reap_premature_chall() {
  log "incomplete after tar — clear SCP/LAUNCHED and reap premature lean by PID"
  ssh "${SSH_OPTS[@]}" -p "$GOLD_PORT" "root@$GOLD_HOST" bash -s <<'REMOTE'
set -euo pipefail
reap() {
  local pid=$1
  [[ "$pid" =~ ^[0-9]+$ ]] || return 0
  kill -0 "$pid" 2>/dev/null || return 0
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 15); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -9 "$pid" 2>/dev/null || true
}
for f in /root/logs/p3693_r653_lean_outer.pid /root/logs/vllm_chall_r653.pid; do
  [[ -f "$f" ]] && reap "$(cat "$f" 2>/dev/null || true)"
done
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  reap "$pid"
done < <(ss -lptn 'sport = :8003' 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u)
rm -f /root/logs/r653_scp_ready.done /root/logs/r653_chall_n80_launched.p3693
REMOTE
}

if [[ ${#missing[@]} -eq 0 || -z "${missing[0]:-}" ]]; then
  n=$(ssh "${SSH_OPTS[@]}" -p "$GOLD_PORT" "root@$GOLD_HOST" \
    "ls $DST/model-*-of-*.safetensors 2>/dev/null | wc -l")
  if [[ "$n" -ge 16 ]] && ssh "${SSH_OPTS[@]}" -p "$GOLD_PORT" "root@$GOLD_HOST" \
      "test -f $DST/config.json"; then
    ssh "${SSH_OPTS[@]}" -p "$GOLD_PORT" "root@$GOLD_HOST" \
      "date -u +%Y-%m-%dT%H:%M:%SZ > $SCP_DONE && echo OK shards=$n && du -sh $DST"
    log "already complete — wrote $SCP_DONE"
    exit 0
  fi
  log "FATAL empty missing list but incomplete"
  exit 1
fi

log "missing (${#missing[@]}): ${missing[*]}"
reap_premature_chall

for f in "${missing[@]}"; do
  ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" "test -f $SRC/$f" \
    || { log "FATAL source missing $f"; exit 1; }
done

MISS_FILE=$(mktemp)
printf '%s\n' "${missing[@]}" >"$MISS_FILE"
scp "${SSH_OPTS[@]}" -P "$BRAVE_PORT" "$MISS_FILE" "root@$BRAVE_HOST:/tmp/r653_missing_p3696.txt"
rm -f "$MISS_FILE"
log "begin patch tar of missing files brave→golden"
set +e
ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" \
  "cd $SRC && tar cf - -T /tmp/r653_missing_p3696.txt" \
  | ssh "${SSH_OPTS[@]}" -p "$GOLD_PORT" "root@$GOLD_HOST" \
    "cd $DST && tar xf -"
rc_b=${PIPESTATUS[0]}
rc_g=${PIPESTATUS[1]}
set -e
log "patch pipe exit brave=$rc_b golden=$rc_g"
[[ "$rc_b" -eq 0 && "$rc_g" -eq 0 ]] || { log "FATAL patch pipe failed"; exit 1; }

n=$(ssh "${SSH_OPTS[@]}" -p "$GOLD_PORT" "root@$GOLD_HOST" \
  "ls $DST/model-*-of-*.safetensors 2>/dev/null | wc -l")
has_cfg=$(ssh "${SSH_OPTS[@]}" -p "$GOLD_PORT" "root@$GOLD_HOST" \
  "test -f $DST/config.json && echo yes || echo no")
s4=$(ssh "${SSH_OPTS[@]}" -p "$GOLD_PORT" "root@$GOLD_HOST" \
  "stat -c%s $DST/model-00004-of-00016.safetensors 2>/dev/null || echo 0")
log "post-patch shards=$n config=$has_cfg shard4_bytes=$s4"
[[ "$n" -ge 16 ]] || { log "FATAL still <16 shards"; exit 1; }
[[ "$has_cfg" == yes ]] || { log "FATAL still no config.json"; exit 1; }
[[ "$s4" -ge 3900000000 ]] || { log "FATAL shard4 still truncated ($s4)"; exit 1; }

# Re-arm waiter if we cleared LAUNCHED (it may have exited after premature launch)
ssh "${SSH_OPTS[@]}" -p "$GOLD_PORT" "root@$GOLD_HOST" bash -s <<'REMOTE'
set -euo pipefail
WAIT=/root/mining_src/r653-chall/wait_r653_scp_king_then_chall_p3693.sh
if [[ ! -f /root/logs/r653_chall_n80_launched.p3693 ]]; then
  # kill stale waiter by pidfile only
  if [[ -f /root/logs/p3693_r653_wait.pid ]]; then
    wpid=$(cat /root/logs/p3693_r653_wait.pid 2>/dev/null || true)
    if [[ "$wpid" =~ ^[0-9]+$ ]] && kill -0 "$wpid" 2>/dev/null; then
      : # still running — leave it
    else
      nohup bash "$WAIT" >/root/logs/p3693_r653_wait.nohup 2>&1 &
      echo $! >/root/logs/p3693_r653_wait.pid
      echo "waiter_rearmed pid=$(cat /root/logs/p3693_r653_wait.pid)"
    fi
  else
    nohup bash "$WAIT" >/root/logs/p3693_r653_wait.nohup 2>&1 &
    echo $! >/root/logs/p3693_r653_wait.pid
    echo "waiter_started pid=$(cat /root/logs/p3693_r653_wait.pid)"
  fi
fi
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r653_scp_ready.done
echo SCP_READY_REPAIR shards=$(ls /tmp/r653_merged/model-*-of-*.safetensors | wc -l)
du -sh /tmp/r653_merged
REMOTE
log "DONE wrote $SCP_DONE — waiter should launch v4 chall+n80"
exit 0
