#!/usr/bin/env bash
# p4351: R1238+R1239 REFUTE vs reign36 — exact-PID reap challs; leave T/K.
# Never pkill -f.
set -euo pipefail
LOG=/root/logs/reap_r1238_r1239_refute_p4351.log
exec >>"$LOG" 2>&1
echo "[p4351-reap] $(date -u +%Y-%m-%dT%H:%M:%SZ) start"

stop_pid() {
  local pid=$1 why=${2:-}
  [[ "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    echo "[p4351-reap] kill pid=$pid ($why)"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null || true
  fi
}

for pf in /root/logs/vllm_chall_r1238.pid /root/logs/vllm_chall_r1239.pid \
          /root/logs/r1238_sim_wvk7.pid /root/logs/r1239_sim_wvk7.pid \
          /root/logs/p4349_r1238_outer.pid /root/logs/p4349_r1239_outer.pid; do
  if [[ -f "$pf" ]]; then
    stop_pid "$(cat "$pf" 2>/dev/null || true)" "pidfile $pf"
    rm -f "$pf"
  fi
done

while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  cmd=$(tr '\0' ' ' </proc/"$pid"/cmdline 2>/dev/null || true)
  if echo "$cmd" | grep -qE '/tmp/r1238_merged|/tmp/r1239_merged'; then
    stop_pid "$pid" "chall merged argv"
  fi
done < <(ps -eo pid=,args= | awk '/vllm serve / && !/awk/ {print $1}')

echo "[p4351-reap] $(date -u +%Y-%m-%dT%H:%M:%SZ) DONE"
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/reap_r1238_r1239_refute_p4351.done
