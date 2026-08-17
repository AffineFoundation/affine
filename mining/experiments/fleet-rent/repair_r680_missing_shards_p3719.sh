#!/usr/bin/env bash
# p3719: R680 zesty→golden post-tar size-check repair (same hole class as R653/R651).
# Wait for live tar to exit, verify ALL 16 shards vs zesty source bytes, patch missing.
# Never pkill -f. Detect tar via `pgrep -c -f '^tar xf'` (no r680_merged in that argv).
set -euo pipefail

ROOT=/home/const/subnet120/mining
LOGDIR=$ROOT/experiments/fleet-rent/logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/repair_r680_missing_shards_p3719.log
: >"$LOG"
log() { echo "[p3719-r680-repair] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=45 -o BatchMode=yes
  -o ServerAliveInterval=15 -o ServerAliveCountMax=40 -o TCPKeepAlive=yes)
ZESTY_HOST=86.38.182.95
ZESTY_PORT=20299
GOLD_HOST=38.127.229.127
GOLD_PORT=40299
SRC=/tmp/r680_merged
DST=/tmp/r680_merged
SCP_DONE=/root/logs/r680_scp_ready.done
LAUNCHED=/root/logs/r680_chall_n80_launched.p3719

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

log "START — wait for live R680 tar recv to exit, then size-check+patch"

for i in $(seq 1 720); do
  alive=$(ssh "${SSH_OPTS[@]}" -p "$GOLD_PORT" "root@$GOLD_HOST" \
    'pgrep -c -f "^tar xf" 2>/dev/null || true' 2>/dev/null || echo 0)
  zesty_tar=$(ssh "${SSH_OPTS[@]}" -p "$ZESTY_PORT" "root@$ZESTY_HOST" \
    "ps aux | grep '[t]ar cf - r680_merged' | wc -l" 2>/dev/null || echo 0)
  n=$(ssh "${SSH_OPTS[@]}" -p "$GOLD_PORT" "root@$GOLD_HOST" \
    "ls $DST/model-*-of-*.safetensors 2>/dev/null | wc -l" 2>/dev/null || echo 0)
  sz=$(ssh "${SSH_OPTS[@]}" -p "$GOLD_PORT" "root@$GOLD_HOST" \
    "du -sh $DST 2>/dev/null | awk '{print \$1}'" 2>/dev/null || echo '?')
  [[ $((i % 3)) -eq 1 ]] && log "poll i=$i tar_xf=$alive zesty_tar=$zesty_tar shards=$n size=$sz"
  if [[ "${alive:-0}" -eq 0 && "${zesty_tar:-0}" -eq 0 && "${n:-0}" -ge 1 ]]; then
    # also wait until host_relay finished or SCP_READY exists / pipe gone
    relay=$(pgrep -c -f 'host_relay_r680_zesty_to_golden_p3719' 2>/dev/null || true)
    if [[ "${relay:-0}" -eq 0 ]] || ssh "${SSH_OPTS[@]}" -p "$GOLD_PORT" "root@$GOLD_HOST" \
      "test -f $SCP_DONE && echo ready || echo no" 2>/dev/null | grep -q ready; then
      log "tar send+recv gone (or SCP_READY) — begin completeness check"
      break
    fi
  fi
  if [[ "$i" -eq 720 ]]; then
    log "FATAL timeout waiting for tar to exit"
    exit 1
  fi
  sleep 20
done

mapfile -t missing < <(ssh "${SSH_OPTS[@]}" -p "$GOLD_PORT" "root@$GOLD_HOST" bash -s <<'REMOTE'
set -euo pipefail
DST=/tmp/r680_merged
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
for f in "${!EXP[@]}"; do
  path="$DST/$f"
  if [[ ! -f "$path" ]]; then
    echo "MISSING $f"
    continue
  fi
  sz=$(stat -c%s "$path")
  if [[ "$sz" -ne "${EXP[$f]}" ]]; then
    echo "TRUNC $f have=$sz want=${EXP[$f]}"
  fi
done
REMOTE
)

if [[ ${#missing[@]} -eq 0 ]]; then
  log "all 16 shards exact — ensure SCP_READY"
  ssh "${SSH_OPTS[@]}" -p "$GOLD_PORT" "root@$GOLD_HOST" \
    "n=\$(ls $DST/model-*-of-*.safetensors | wc -l); test \"\$n\" -ge 16 && test -f $DST/config.json && date -u +%Y-%m-%dT%H:%M:%SZ > $SCP_DONE && echo OK shards=\$n"
  log "DONE already-complete"
  exit 0
fi

log "need repair: ${missing[*]}"
# If chall already launched with exact merge, do not clobber
if ssh "${SSH_OPTS[@]}" -p "$GOLD_PORT" "root@$GOLD_HOST" "test -f $LAUNCHED && echo yes || echo no" | grep -q yes; then
  log "CHALL already launched — skip patch (manual inspect)"
  exit 0
fi

for line in "${missing[@]}"; do
  f=$(echo "$line" | awk '{print $2}')
  [[ -n "$f" ]] || continue
  log "patch $f from zesty"
  ssh "${SSH_OPTS[@]}" -p "$ZESTY_PORT" "root@$ZESTY_HOST" "cat $SRC/$f" \
    | ssh "${SSH_OPTS[@]}" -p "$GOLD_PORT" "root@$GOLD_HOST" "cat > $DST/$f.tmp && mv $DST/$f.tmp $DST/$f"
done

ssh "${SSH_OPTS[@]}" -p "$GOLD_PORT" "root@$GOLD_HOST" bash -s <<'REMOTE'
set -euo pipefail
DST=/tmp/r680_merged
SCP_DONE=/root/logs/r680_scp_ready.done
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
bad=0
for f in "${!EXP[@]}"; do
  path="$DST/$f"
  [[ -f "$path" ]] || { echo "STILL_MISSING $f"; bad=1; continue; }
  sz=$(stat -c%s "$path")
  [[ "$sz" -eq "${EXP[$f]}" ]] || { echo "STILL_TRUNC $f $sz"; bad=1; }
done
[[ "$bad" -eq 0 ]]
n=$(ls "$DST"/model-*-of-*.safetensors | wc -l)
test "$n" -ge 16
test -f "$DST/config.json"
date -u +%Y-%m-%dT%H:%M:%SZ >"$SCP_DONE"
echo "SCP_READY_REPAIRED shards=$n"
du -sh "$DST"
REMOTE

log "DONE repair+SCP_READY"
