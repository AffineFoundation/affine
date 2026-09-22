#!/bin/bash
# Env-backfill driver pod: relaunch the drivers that should be running but are
# not. Thin wrapper — the logic (board-driven source lists, dead-box and
# retired-record skips) is rollouts/scripts/backfill_relaunch.py; see its
# docstring. Run by the post-start hook and pipeline-health's self-heal.
#   backfill_relaunch.sh [--dry-run]
set -uo pipefail
cd /root/rollouts 2>/dev/null || true
[ -f /root/rollouts/.rollouts_env ] && source /root/rollouts/.rollouts_env
export PYTHONPATH=/root/affine:/root/rollouts
PY=/root/venv/bin/python; [ -x "$PY" ] || PY=python3
exec "$PY" /root/rollouts/scripts/backfill_relaunch.py "$@"
