#!/usr/bin/env bash
# p3696: R655 dest mid-SCP has truncated model-00001 + missing odd shards
# (same hole class as R651/R634). Wait for live R252→lunar tar to exit,
# verify ALL 16 shards against source byte sizes, patch only missing/truncated,
# then write r655_scp_ready.done. Never dual-pipe (R653 is brave→golden).
# Never pkill -f. Detect tar via `pgrep -c -f '^tar xf'` (no r655_merged in argv).
set -euo pipefail

ROOT=/home/const/subnet120/mining
LOGDIR=$ROOT/experiments/fleet-rent/logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/repair_r655_missing_shards_p3696.log
: >"$LOG"
log() { echo "[p3696-r655-repair] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=45 -o BatchMode=yes
  -o ServerAliveInterval=15 -o ServerAliveCountMax=40 -o TCPKeepAlive=yes)
SRC_HOST=95.133.252.28
SRC_PORT=40299
LUNAR_HOST=150.136.46.118
LUNAR_PORT=20299
SRC=/tmp/r655_merged
DST=/tmp/r655_merged
SCP_DONE=/root/logs/r655_scp_ready.done

log "START — wait for live R655 tar recv to exit, then size-check+patch"

for i in $(seq 1 720); do
  alive=$(ssh "${SSH_OPTS[@]}" -p "$LUNAR_PORT" "root@$LUNAR_HOST" \
    'pgrep -c -f "^tar xf" 2>/dev/null || true' 2>/dev/null || echo 0)
  src_tar=$(ssh "${SSH_OPTS[@]}" -p "$SRC_PORT" "root@$SRC_HOST" \
    "ps aux | grep '[t]ar cf - r655_merged' | wc -l" 2>/dev/null || echo 0)
  n=$(ssh "${SSH_OPTS[@]}" -p "$LUNAR_PORT" "root@$LUNAR_HOST" \
    "ls $DST/model-*-of-*.safetensors 2>/dev/null | wc -l" 2>/dev/null || echo 0)
  sz=$(ssh "${SSH_OPTS[@]}" -p "$LUNAR_PORT" "root@$LUNAR_HOST" \
    "du -sh $DST 2>/dev/null | awk '{print \$1}'" 2>/dev/null || echo '?')
  [[ $((i % 3)) -eq 1 ]] && log "poll i=$i tar_xf=$alive src_tar=$src_tar shards=$n size=$sz"
  if [[ "${alive:-0}" -eq 0 && "${src_tar:-0}" -eq 0 ]]; then
    log "tar send+recv gone — begin completeness check"
    break
  fi
  if [[ "$i" -eq 720 ]]; then
    log "FATAL timeout waiting for tar to exit"
    exit 1
  fi
  sleep 20
done

mapfile -t missing < <(ssh "${SSH_OPTS[@]}" -p "$LUNAR_PORT" "root@$LUNAR_HOST" bash -s <<'REMOTE'
set -euo pipefail
DST=/tmp/r655_merged
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
  want=${EXP[$f]}
  min=$(( want * 995 / 1000 ))
  [[ "$sz" -lt "$min" ]] && echo "$f"
done
REMOTE
)

mapfile -t missing < <(printf '%s\n' "${missing[@]:-}" | awk 'NF && !seen[$0]++')

reap_premature_chall() {
  log "incomplete after tar — clear SCP/LAUNCHED and reap premature lean by PID"
  ssh "${SSH_OPTS[@]}" -p "$LUNAR_PORT" "root@$LUNAR_HOST" bash -s <<'REMOTE'
set -euo pipefail
reap() {
  local pid=$1
  [[ "$pid" =~ ^[0-9]+$ ]] || return 0
  kill -0 "$pid" 2>/dev/null || return 0
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 15); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -9 "$pid" 2>/dev/null || true
}
for f in /root/logs/p3688_r655_lean_outer.pid /root/logs/vllm_chall_r655.pid; do
  [[ -f "$f" ]] && reap "$(cat "$f" 2>/dev/null || true)"
done
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  reap "$pid"
done < <(ss -lptn 'sport = :8003' 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u)
rm -f /root/logs/r655_scp_ready.done /root/logs/r655_chall_n80_launched.p3688
REMOTE
}

if [[ ${#missing[@]} -eq 0 || -z "${missing[0]:-}" ]]; then
  n=$(ssh "${SSH_OPTS[@]}" -p "$LUNAR_PORT" "root@$LUNAR_HOST" \
    "ls $DST/model-*-of-*.safetensors 2>/dev/null | wc -l")
  if [[ "$n" -ge 16 ]] && ssh "${SSH_OPTS[@]}" -p "$LUNAR_PORT" "root@$LUNAR_HOST" \
      "test -f $DST/config.json"; then
    ssh "${SSH_OPTS[@]}" -p "$LUNAR_PORT" "root@$LUNAR_HOST" \
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
  ssh "${SSH_OPTS[@]}" -p "$SRC_PORT" "root@$SRC_HOST" "test -f $SRC/$f" \
    || { log "FATAL source missing $f"; exit 1; }
done

MISS_FILE=$(mktemp)
printf '%s\n' "${missing[@]}" >"$MISS_FILE"
scp "${SSH_OPTS[@]}" -P "$SRC_PORT" "$MISS_FILE" "root@$SRC_HOST:/tmp/r655_missing_p3696.txt"
rm -f "$MISS_FILE"
log "begin patch tar of missing files R252→lunar"
set +e
ssh "${SSH_OPTS[@]}" -p "$SRC_PORT" "root@$SRC_HOST" \
  "cd $SRC && tar cf - -T /tmp/r655_missing_p3696.txt" \
  | ssh "${SSH_OPTS[@]}" -p "$LUNAR_PORT" "root@$LUNAR_HOST" \
    "cd $DST && tar xf -"
rc_s=${PIPESTATUS[0]}
rc_l=${PIPESTATUS[1]}
set -e
log "patch pipe exit src=$rc_s lunar=$rc_l"
[[ "$rc_s" -eq 0 && "$rc_l" -eq 0 ]] || { log "FATAL patch pipe failed"; exit 1; }

n=$(ssh "${SSH_OPTS[@]}" -p "$LUNAR_PORT" "root@$LUNAR_HOST" \
  "ls $DST/model-*-of-*.safetensors 2>/dev/null | wc -l")
has_cfg=$(ssh "${SSH_OPTS[@]}" -p "$LUNAR_PORT" "root@$LUNAR_HOST" \
  "test -f $DST/config.json && echo yes || echo no")
s1=$(ssh "${SSH_OPTS[@]}" -p "$LUNAR_PORT" "root@$LUNAR_HOST" \
  "stat -c%s $DST/model-00001-of-00016.safetensors 2>/dev/null || echo 0")
log "post-patch shards=$n config=$has_cfg shard1_bytes=$s1"
[[ "$n" -ge 16 ]] || { log "FATAL still <16 shards"; exit 1; }
[[ "$has_cfg" == yes ]] || { log "FATAL still no config.json"; exit 1; }
[[ "$s1" -ge 4200000000 ]] || { log "FATAL shard1 still truncated ($s1)"; exit 1; }

ssh "${SSH_OPTS[@]}" -p "$LUNAR_PORT" "root@$LUNAR_HOST" bash -s <<'REMOTE'
set -euo pipefail
WAIT=/root/mining_src/r655-chall/wait_r655_scp_king_then_chall_p3688.sh
if [[ ! -f /root/logs/r655_chall_n80_launched.p3688 ]]; then
  if [[ -f /root/logs/p3688_r655_wait.pid ]]; then
    wpid=$(cat /root/logs/p3688_r655_wait.pid 2>/dev/null || true)
    if [[ "$wpid" =~ ^[0-9]+$ ]] && kill -0 "$wpid" 2>/dev/null; then
      :
    else
      nohup bash "$WAIT" >/root/logs/p3688_r655_wait.nohup 2>&1 &
      echo $! >/root/logs/p3688_r655_wait.pid
      echo "waiter_rearmed pid=$(cat /root/logs/p3688_r655_wait.pid)"
    fi
  else
    nohup bash "$WAIT" >/root/logs/p3688_r655_wait.nohup 2>&1 &
    echo $! >/root/logs/p3688_r655_wait.pid
    echo "waiter_started pid=$(cat /root/logs/p3688_r655_wait.pid)"
  fi
fi
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r655_scp_ready.done
echo SCP_READY_REPAIR shards=$(ls /tmp/r655_merged/model-*-of-*.safetensors | wc -l)
du -sh /tmp/r655_merged
REMOTE
log "DONE wrote $SCP_DONE — waiter should launch v4 chall+n80"
exit 0
