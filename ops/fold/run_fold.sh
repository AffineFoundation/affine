#!/bin/bash
# pm2 wrapper for the fold (affine-corpus-refresh). Cron-driven: pm2 starts
# it on the schedule, it runs one fold through ops/fold/fold_wrap.py (start /
# exit / traceback -> affine/state/fold/last_run.json, one retry on a
# transient failure, Discord page on any final non-zero exit) and exits.
#
#   pm2 start ops/fold/run_fold.sh --name affine-corpus-refresh --interpreter bash \
#       --cron "0 */6 * * *" --no-autorestart
# Extra arguments are passed to ops/corpus_build.py unchanged.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# the fold needs the whole operator env (R2 / DATA_R2 / HIPPIUS / Discord …)
export PM2_ENV_ALL=1
source "$REPO/ops/health/pm2_env.sh"
unset PM2_ENV_ALL
export PYTHONUNBUFFERED=1
cd "$REPO"
exec "$REPO/.venv/bin/python" ops/fold/fold_wrap.py -- "$@"
