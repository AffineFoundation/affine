#!/usr/bin/env bash
# p3873: p3871 stamp path bug — hypo=r784 writes /root/logs/r${hypo}_…
# → rr784_scp_ready.done, but waiters poll r784_scp_ready.done. Size-verify
# all 16+vis+config vs brave, then write the CORRECT stamps. Also copy any
# accidental rr* stamps. Never pkill -f. Leave live pipes / R800/R801 alone.
set -euo pipefail
ROOT=/home/const/subnet120/mining
LOGDIR=$ROOT/experiments/fleet-rent/logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/host_relay_r784_r783_fix_stamp_p3873.log
: >"$LOG"
log() { echo "[p3873-stampfix] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=45 -o BatchMode=yes
  -o ServerAliveInterval=15 -o ServerAliveCountMax=40 -o TCPKeepAlive=yes)
BRAVE_HOST=18.118.83.97
BRAVE_PORT=40127
CROWN_HOST=95.133.253.90
CROWN_PORT=40099

FILES=(
  model-00001-of-00016.safetensors
  model-00002-of-00016.safetensors
  model-00003-of-00016.safetensors
  model-00004-of-00016.safetensors
  model-00005-of-00016.safetensors
  model-00006-of-00016.safetensors
  model-00007-of-00016.safetensors
  model-00008-of-00016.safetensors
  model-00009-of-00016.safetensors
  model-00010-of-00016.safetensors
  model-00011-of-00016.safetensors
  model-00012-of-00016.safetensors
  model-00013-of-00016.safetensors
  model-00014-of-00016.safetensors
  model-00015-of-00016.safetensors
  model-00016-of-00016.safetensors
  model-visual-restored.safetensors
)

src_size() {
  local hypo=$1 f=$2 out
  out=$(ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" \
    "stat -c%s /tmp/${hypo}_merged/$f 2>/dev/null || echo 0" 2>/dev/null || echo 0)
  echo "${out//[^0-9]/}"
}

dest_size() {
  local hypo=$1 f=$2 out
  out=$(ssh "${SSH_OPTS[@]}" -p "$CROWN_PORT" "root@$CROWN_HOST" \
    "stat -c%s /tmp/${hypo}_merged/$f 2>/dev/null || echo 0" 2>/dev/null || echo 0)
  echo "${out//[^0-9]/}"
}

tmp_count() {
  local hypo=$1 out
  out=$(ssh "${SSH_OPTS[@]}" -p "$CROWN_PORT" "root@$CROWN_HOST" \
    "ls /tmp/${hypo}_merged/*.tmp 2>/dev/null | wc -l" 2>/dev/null || echo 99)
  echo "${out//[^0-9]/}"
}

stamp_hypo() {
  local hypo=$1
  # Write CORRECT path (${hypo}_scp_ready.done). Also repair accidental rr* stamp.
  ssh "${SSH_OPTS[@]}" -p "$CROWN_PORT" "root@$CROWN_HOST" \
    "hypo=$hypo bash -s" <<'REMOTE'
set -euo pipefail
mkdir -p /root/logs
ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "$ts" > "/root/logs/${hypo}_scp_ready.done"
# If p3871 wrote r${hypo}=rr784 / rr783, mirror it so nothing else waits on typo
if [[ -f "/root/logs/r${hypo}_scp_ready.done" ]]; then
  cp -f "/root/logs/${hypo}_scp_ready.done" "/root/logs/r${hypo}_scp_ready.done" || true
fi
n=$(ls /tmp/${hypo}_merged/model-*-of-*.safetensors 2>/dev/null | wc -l)
vis=0; [[ -f /tmp/${hypo}_merged/model-visual-restored.safetensors ]] && vis=1
cfg=0; [[ -f /tmp/${hypo}_merged/config.json ]] && cfg=1
echo "stamped:${hypo}:$ts:n=$n:vis=$vis:cfg=$cfg:$(du -sh /tmp/${hypo}_merged | awk '{print $1}')"
REMOTE
}

verify_hypo() {
  local hypo=$1
  local bad=0 f want have tmps cfg
  for f in "${FILES[@]}"; do
    want=$(src_size "$hypo" "$f"); want=${want:-0}
    if [[ "$want" -le 0 ]]; then
      echo "nosrc:$f"
      return 1
    fi
    have=$(dest_size "$hypo" "$f"); have=${have:-0}
    if [[ "$have" != "$want" ]]; then
      echo "mismatch:$f:$have/$want"
      return 1
    fi
  done
  tmps=$(tmp_count "$hypo"); tmps=${tmps:-99}
  if [[ "$tmps" -ne 0 ]]; then
    echo "tmps:$tmps"
    return 1
  fi
  cfg=$(ssh "${SSH_OPTS[@]}" -p "$CROWN_PORT" "root@$CROWN_HOST" \
    "test -f /tmp/${hypo}_merged/config.json && echo 1 || echo 0" 2>/dev/null || echo 0)
  if [[ "$cfg" != "1" ]]; then
    echo "nocfg"
    return 1
  fi
  echo "ok"
  return 0
}

log "armed: poll crown size-verify → correct r784/r783 stamps (p3871 path bug)"

done_r784=0
done_r783=0
for i in $(seq 1 900); do
  if [[ "$done_r784" -eq 0 ]]; then
    v=$(verify_hypo r784 || true)
    if [[ "$v" == ok ]]; then
      out=$(stamp_hypo r784)
      log "R784 SIZE_OK $out"
      done_r784=1
    else
      (( i % 12 == 0 )) && log "R784 wait iter=$i detail=$v"
    fi
  fi
  if [[ "$done_r783" -eq 0 ]]; then
    v=$(verify_hypo r783 || true)
    if [[ "$v" == ok ]]; then
      out=$(stamp_hypo r783)
      log "R783 SIZE_OK $out"
      done_r783=1
    else
      (( i % 12 == 0 )) && log "R783 wait iter=$i detail=$v"
    fi
  fi
  if [[ "$done_r784" -eq 1 && "$done_r783" -eq 1 ]]; then
    log "DONE both stamps correct — waiters should arm lean challs"
    date -u +%Y-%m-%dT%H:%M:%SZ >"$LOGDIR/host_relay_r784_r783_fix_stamp_p3873.done"
    exit 0
  fi
  sleep 10
done
log "FATAL timeout done_r784=$done_r784 done_r783=$done_r783"
exit 1
