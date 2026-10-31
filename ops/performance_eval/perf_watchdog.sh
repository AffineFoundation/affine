#!/bin/bash
# Watchdog for affine-perf-eval (run by affine-perf-watchdog.timer every 5 min): restart the service if it is not active or its
# heartbeat is older than 10 min. Touches nothing else.
HB=/home/const/subnet120-rewrite/state/root-audits/20261031-performance-eval/state/heartbeat
L=/home/const/subnet120-rewrite/state/root-audits/20261031-performance-eval/state/watchdog.log
now=$(date +%s); hb=$(cat $HB 2>/dev/null | cut -d. -f1); hb=${hb:-0}
if ! systemctl --user is-active --quiet affine-perf-eval || [ $((now-hb)) -gt 600 ]; then
  echo "$(TZ=America/Costa_Rica date '+%F %T CR') restart (active=$(systemctl --user is-active affine-perf-eval) hb_age=$((now-hb))s)" >> $L
  systemctl --user restart affine-perf-eval
fi
