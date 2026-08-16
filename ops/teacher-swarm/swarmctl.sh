#!/bin/bash
# Pidfile-based control for the swarm manager + router (no pkill footguns).
# Usage: ./swarmctl.sh {start|stop|restart|status} [manager|router]
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p state
VENV=/home/const/subnet120/.venv/bin/python

start_one() {  # name cmd...
  local name=$1; shift
  local pidf="state/$name.pid"
  if [ -f "$pidf" ] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
    echo "$name already running (pid $(cat "$pidf"))"; return 0
  fi
  setsid nohup "$@" >> "state/$name.log" 2>&1 < /dev/null &
  echo $! > "$pidf"
  echo "$name started (pid $!)"
}

stop_one() {
  local name=$1
  local pidf="state/$name.pid"
  if [ -f "$pidf" ] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
    # setsid made the pid a group leader; group-kill takes children (ssh) too.
    kill -- "-$(cat "$pidf")" 2>/dev/null || kill "$(cat "$pidf")"
    echo "$name stopped"
  else
    echo "$name not running"
  fi
  rm -f "$pidf"
}

status_one() {
  local name=$1
  local pidf="state/$name.pid"
  if [ -f "$pidf" ] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
    echo "$name running (pid $(cat "$pidf"))"
  else
    echo "$name stopped"
  fi
}

cmd=${1:-status}
what=${2:-all}

run() {  # action target
  case $2 in
    manager) case $1 in
        start) start_one manager "$VENV" manager.py --interval 30 ;;
        stop) stop_one manager ;;
        status) status_one manager ;;
      esac ;;
    router) case $1 in
        start) start_one router "$VENV" router.py --port 9100 ;;
        stop) stop_one router ;;
        status) status_one router ;;
      esac ;;
    tunnel) case $1 in
        start) start_one tunnel bash tunnel_eval.sh ;;
        stop) stop_one tunnel ;;
        status) status_one tunnel ;;
      esac ;;
  esac
}

targets=()
if [ "$what" = all ]; then targets=(manager router tunnel); else targets=("$what"); fi
for t in "${targets[@]}"; do
  case $cmd in
    start|stop|status) run "$cmd" "$t" ;;
    restart) run stop "$t"; sleep 1; run start "$t" ;;
    *) echo "usage: $0 {start|stop|restart|status} [manager|router|all]"; exit 2 ;;
  esac
done
