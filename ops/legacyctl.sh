#!/usr/bin/env bash
# SN120 legacy-winner weight pin toggle. Runs ops/legacy_weights.py in the
# background, which splits the weight vector equally across the previous
# winners of the old Affine mechanism every 5 minutes until turned off.
#
#   ./ops/legacyctl.sh on        start pinning weights on legacy winners
#   ./ops/legacyctl.sh off       stop pinning
#   ./ops/legacyctl.sh status    running? + last log lines
#
# Log: ops/legacy.log   Pidfile: ops/legacy.pid
# Do NOT run at the same time as burnctl.sh or the production validator.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIDFILE="$ROOT/ops/legacy.pid"
LOGFILE="$ROOT/ops/legacy.log"
BURN_PIDFILE="$ROOT/ops/burn.pid"

running() { [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }

case "${1:-status}" in
  on)
    if running; then
      echo "already ON (pid $(cat "$PIDFILE"))"
      exit 0
    fi
    if [[ -f "$BURN_PIDFILE" ]] && kill -0 "$(cat "$BURN_PIDFILE")" 2>/dev/null; then
      echo "refusing: burn loop is ON (pid $(cat "$BURN_PIDFILE")) — run ./ops/burnctl.sh off first" >&2
      exit 1
    fi
    # shellcheck source=/dev/null
    source "$ROOT/.venv/bin/activate"
    nohup python "$ROOT/ops/legacy_weights.py" >>"$LOGFILE" 2>&1 &
    echo $! >"$PIDFILE"
    echo "legacy-winner loop ON (pid $(cat "$PIDFILE")) — log: $LOGFILE"
    ;;
  off)
    if running; then
      kill "$(cat "$PIDFILE")"
      rm -f "$PIDFILE"
      echo "legacy-winner loop OFF."
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
