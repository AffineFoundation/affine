#!/bin/bash
# pm2 wrappers for the coverage tooling (operator directive 2026-09-15: the
# affine.io/#kings tables must always be complete).
#
#   run.sh queue     benchmark backfill queue loop (bench_queue.py loop)
#                    pm2 start ops/coverage/run.sh --name affine-coverage-bench --interpreter bash -- queue
#   run.sh nightly   coverage check -> state/coverage.json + one Discord line in the
#                    private Arbos channel; pm2 cron entry (see notes in coverage.py)
#                    pm2 start ops/coverage/run.sh --name affine-coverage-nightly --interpreter bash \
#                        --cron-restart "30 6 * * *" --no-autorestart -- nightly
#
# Env comes from the box snapshot through ops/benchsuite/env.sh (LIUM_API_KEY,
# PRIME_API_KEY, HF_TOKEN, DATA_R2_*, DISCORD_BOT_TOKEN_ARBOS_BITTENSOR, OP token).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="${COVERAGE_PYTHON:-$REPO/.venv/bin/python}"
# shellcheck disable=SC1091
source "$REPO/ops/benchsuite/env.sh"
cd "$HERE"
case "${1:-}" in
  queue)   exec "$PY" bench_queue.py loop --interval "${COVERAGE_QUEUE_INTERVAL_S:-300}" ;;
  nightly) "$PY" coverage.py --json "$HERE/state/coverage.json" --markdown "$HERE/state/coverage.md" --post
           exec "$PY" autofill.py ;;             # the check LAUNCHES what it finds missing (2026-09-17)
  autofill) "$PY" coverage.py --json "$HERE/state/coverage.json" --markdown "$HERE/state/coverage.md"
           exec "$PY" autofill.py ;;             # every 4 h (pm2 cron): new cards / new columns get filled the same day
  check)   exec "$PY" coverage.py --json "$HERE/state/coverage.json" --markdown "$HERE/state/coverage.md" ;;
  *) echo "usage: run.sh queue|nightly|autofill|check" >&2; exit 2 ;;
esac
