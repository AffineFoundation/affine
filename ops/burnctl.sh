#!/usr/bin/env bash
# SN120 emission kill-switch toggle. Runs ops/burn_weights.py in the
# background, which pins 100% of the weight vector on the burn UID every
# 5 minutes until turned off.
#
#   ./ops/burnctl.sh on        start burning miner emission
#   ./ops/burnctl.sh off       stop burning (then restart the validator
#                              to resume normal king-chain weights)
#   ./ops/burnctl.sh status    running? + last log lines
#
# Log: ops/burn.log   Pidfile: ops/burn.pid
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIDFILE="$ROOT/ops/burn.pid"
LOGFILE="$ROOT/ops/burn.log"

running() { [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }

case "${1:-status}" in
  on)
    if running; then
      echo "already ON (pid $(cat "$PIDFILE"))"
      exit 0
    fi
    # shellcheck source=/dev/null
    source "$ROOT/.venv/bin/activate"
    nohup python "$ROOT/ops/burn_weights.py" >>"$LOGFILE" 2>&1 &
    echo $! >"$PIDFILE"
    echo "burn loop ON (pid $(cat "$PIDFILE")) — weights pinned to burn uid; log: $LOGFILE"
    ;;
  off)
    if running; then
      kill "$(cat "$PIDFILE")"
      rm -f "$PIDFILE"
      echo "burn loop OFF. Restart the validator to resume normal weights."
    else
      rm -f "$PIDFILE"
      echo "not running"
    fi
    ;;
  status)
    if running; then
      echo "ON (pid $(cat "$PIDFILE"))"
    else
      echo "OFF"
    fi
    [[ -f "$LOGFILE" ]] && tail -n 5 "$LOGFILE"
    ;;
  *)
    echo "usage: $0 on|off|status" >&2
    exit 1
    ;;
esac
