#!/usr/bin/env bash
# p3706: post-tar size-check repair for R675 R252→lunar (same hole class as R655/R651).
# Wait for live tar to exit, compare ALL 16 shards to source bytes, patch missing/truncated,
# then write r675_scp_ready.done. Never dual-pipe. Never pkill -f.
set -euo pipefail

ROOT=/home/const/subnet120/mining
LOGDIR=$ROOT/experiments/fleet-rent/logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/repair_r675_missing_shards_p3706.log
: >"$LOG"
log() { echo "[p3706-r675-repair] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=45 -o BatchMode=yes
  -o ServerAliveInterval=15 -o ServerAliveCountMax=40 -o TCPKeepAlive=yes)
SRC_HOST=95.133.252.28
SRC_PORT=40299
LUNAR_HOST=150.136.46.118
LUNAR_PORT=20299
DST=/tmp/r675_merged
SCP_DONE=/root/logs/r675_scp_ready.done

log "START — wait for live R675 tar recv to exit, then size-check+patch"
seen_tar=0

for i in $(seq 1 720); do
  alive=$(ssh "${SSH_OPTS[@]}" -p "$LUNAR_PORT" "root@$LUNAR_HOST" \
    'pgrep -c -f "^tar xf" 2>/dev/null || true' 2>/dev/null || echo 0)
  src_tar=$(ssh "${SSH_OPTS[@]}" -p "$SRC_PORT" "root@$SRC_HOST" \
    "ps aux | grep '[t]ar cf - r675_merged' | wc -l" 2>/dev/null || echo 0)
  n=$(ssh "${SSH_OPTS[@]}" -p "$LUNAR_PORT" "root@$LUNAR_HOST" \
    "ls $DST/model-*-of-*.safetensors 2>/dev/null | wc -l" 2>/dev/null || echo 0)
  sz=$(ssh "${SSH_OPTS[@]}" -p "$LUNAR_PORT" "root@$LUNAR_HOST" \
    "du -sh $DST 2>/dev/null | awk '{print \$1}'" 2>/dev/null || echo '?')
  [[ $((i % 3)) -eq 1 ]] && log "poll i=$i tar_xf=$alive src_tar=$src_tar shards=$n size=$sz seen=$seen_tar"
  if [[ "${alive:-0}" -ge 1 || "${src_tar:-0}" -ge 1 ]]; then
    seen_tar=1
  fi
  # only conclude "gone" after we have observed a live tar at least once
  if [[ "$seen_tar" -eq 1 && "${alive:-0}" -eq 0 && "${src_tar:-0}" -eq 0 ]]; then
    log "tar send+recv gone (after seen) — begin completeness check"
    break
  fi
  if [[ "$i" -eq 720 ]]; then
    log "FATAL timeout waiting for tar to exit"
    exit 1
  fi
  sleep 20
done

# Dynamic: list dest files whose size != source size (or missing)
mapfile -t missing < <(python3 - <<'PY'
import subprocess, os
ssh_opts = ["ssh","-o","StrictHostKeyChecking=accept-new","-o","BatchMode=yes","-o","ConnectTimeout=45"]
def run(port, host, cmd):
    return subprocess.check_output(ssh_opts+["-p",str(port),f"root@{host}",cmd], text=True)
src = run(40299, "95.133.252.28",
  "cd /tmp/r675_merged && for f in model-*-of-*.safetensors config.json; do "
  "test -f \"$f\" && echo \"$(stat -c%s \"$f\") $f\"; done")
src_map = {}
for line in src.strip().splitlines():
    sz, name = line.split(None, 1)
    src_map[name] = int(sz)
dst = run(20299, "150.136.46.118",
  "cd /tmp/r675_merged 2>/dev/null && for f in model-*-of-*.safetensors config.json; do "
  "test -f \"$f\" && echo \"$(stat -c%s \"$f\") $f\"; done || true")
dst_map = {}
for line in dst.strip().splitlines():
    parts = line.split(None, 1)
    if len(parts)==2:
        dst_map[parts[1]] = int(parts[0])
for name, sz in sorted(src_map.items()):
    dsz = dst_map.get(name)
    if dsz is None or dsz != sz:
        print(name)
PY
)

if [[ ${#missing[@]} -eq 0 ]]; then
  log "all shards+config match source sizes — write SCP_READY if needed"
  ssh "${SSH_OPTS[@]}" -p "$LUNAR_PORT" "root@$LUNAR_HOST" \
    "n=\$(ls $DST/model-*-of-*.safetensors 2>/dev/null | wc -l); test \"\$n\" -ge 16 && test -f $DST/config.json && date -u +%Y-%m-%dT%H:%M:%SZ > $SCP_DONE && echo OK shards=\$n"
  log "DONE (no repair needed)"
  exit 0
fi

log "need patch: ${missing[*]}"
for f in "${missing[@]}"; do
  log "scp patch $f"
  ssh "${SSH_OPTS[@]}" -p "$SRC_PORT" "root@$SRC_HOST" \
    "cat /tmp/r675_merged/$f" \
    | ssh "${SSH_OPTS[@]}" -p "$LUNAR_PORT" "root@$LUNAR_HOST" \
      "mkdir -p $DST && cat > $DST/$f.tmp && mv $DST/$f.tmp $DST/$f"
done
ssh "${SSH_OPTS[@]}" -p "$LUNAR_PORT" "root@$LUNAR_HOST" \
  "n=\$(ls $DST/model-*-of-*.safetensors 2>/dev/null | wc -l); test \"\$n\" -ge 16 && test -f $DST/config.json && date -u +%Y-%m-%dT%H:%M:%SZ > $SCP_DONE && echo SCP_READY shards=\$n && du -sh $DST"
log "DONE repair+SCP_READY"
