#!/bin/bash
# Watchdog for the performance eval services (affine-perf-watchdog.timer, every 5 min): restart a unit if it is not active or its
# heartbeat is older than 10 min. Touches nothing else. Pod 2's unit is skipped once it has torn its pod down (it then only idles).
S=/home/const/subnet120-rewrite/state/root-audits/20261031-performance-eval/state; L=$S/watchdog.log; now=$(date +%s)
for pair in "affine-perf-eval:calm-orbit-ef" "affine-perf-eval-2:eager-wolf-98"; do
  u=${pair%%:*}; p=${pair##*:}
  [ -f ~/.config/systemd/user/$u.service ] || continue
  systemctl --user is-enabled --quiet $u || continue
  hb=$(cut -d. -f1 $S/heartbeat-$p 2>/dev/null); hb=${hb:-0}
  if ! systemctl --user is-active --quiet $u || [ $((now-hb)) -gt 600 ]; then
    echo "$(TZ=America/Costa_Rica date '+%F %T CR') restart $u (active=$(systemctl --user is-active $u) hb_age=$((now-hb))s)" >> $L
    systemctl --user restart $u
  fi
done
