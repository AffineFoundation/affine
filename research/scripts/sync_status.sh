#!/usr/bin/env bash
# Copy new loop/bench status lines from the pod into the local coordinator log.
# Only real stage and round lines are kept: transient per-turn sampling errors
# would otherwise bury the signal.
set -u
POD_HOST="${POD_HOST:-root@204.9.206.216}"
POD_PORT="${POD_PORT:-40301}"
LOCAL="${LOCAL:-/home/const/subnet120/research/logs/trackA_status.log}"
TMP="$(mktemp)"

timeout 120 ssh -o StrictHostKeyChecking=no -p "$POD_PORT" "$POD_HOST" \
  'cat /root/work/loop_status.log /root/work/bench_wave.log 2>/dev/null' 2>/dev/null \
  | grep -E "valid_rate=|COLLAPSE|RESULT |BENCH |DRIVER |LOOP (init|data|models_loaded|warn|resumed|abort)" \
  | grep -vE "no usable samples|sample_error" | sort -u > "$TMP"

added=0
while IFS= read -r line; do
  [ -z "$line" ] && continue
  if ! grep -Fqx "$line" "$LOCAL" 2>/dev/null; then
    echo "$line" >> "$LOCAL"
    added=$((added + 1))
  fi
done < "$TMP"
rm -f "$TMP"
echo "synced $added new lines; local total $(wc -l < "$LOCAL")"
