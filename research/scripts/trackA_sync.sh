#!/usr/bin/env bash
# Mirrors new lines from the pod's loop_status.log and bench_wave.log into the
# coordinator status log. Line-offset cursors make it append-only and idempotent.
set -u
POD="ssh -p 40301 -o ConnectTimeout=15 root@204.9.206.216"
LOCAL_LOG=/home/const/subnet120/research/logs/trackA_status.log
STATE_DIR=/home/const/subnet120/research/logs/.trackA_sync
mkdir -p "$STATE_DIR"

sync_file() {
  local remote="$1" tag="$2" transform="$3"
  local cursor="$STATE_DIR/$(basename "$remote").offset"
  local off=0
  [ -f "$cursor" ] && off=$(cat "$cursor")
  local total
  total=$($POD "wc -l < $remote" 2>/dev/null) || return 0
  total=$(echo "$total" | tr -dc 0-9)
  [ -z "$total" ] && return 0
  if [ "$total" -lt "$off" ]; then off=0; fi   # remote file rotated/truncated
  if [ "$total" -gt "$off" ]; then
    $POD "tail -n +$((off + 1)) $remote" 2>/dev/null \
      | head -n $((total - off)) \
      | sed -e "$transform" >> "$LOCAL_LOG"
    echo "$total" > "$cursor"
  fi
}

while true; do
  sync_file /root/work/loop_status.log loop 's/ | ROUND / | ARMB ROUND /'
  sync_file /root/work/bench_wave.log bench 's/^/&/'
  sleep 120
done
