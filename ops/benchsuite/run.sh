#!/bin/bash
# pm2 wrapper for the benchmark-suite watcher (watch.py). Loads the frozen env
# snapshot the same way ops/run_validator.sh does (doppler is broken on the
# box): KEY=VALUE lines from ~/.affine-validator.env and the repo .env,
# exported through a shell-safe parser (never `source` them directly).
# Uses PRIME_API_KEY (Prime pods + sandboxes + evals), HF_TOKEN (gated
# benchmark data), DATA_R2_* (publish to research/benchsuite/ on affine-data).
# ~/.affine-op.env (0600) holds OP_SERVICE_ACCOUNT_TOKEN so run_pass.sh can read
# the Docker Hub pull-cap login from the Arbos vault at pod time (op CLI in
# ~/.local/bin; never stored in the repo or the validator env snapshot).
#
#   pm2 start ops/benchsuite/run.sh --name affine-benchsuite --interpreter bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="${BENCHSUITE_PYTHON:-$REPO/.venv/bin/python}"

# shellcheck disable=SC1091
source "$HERE/env.sh"

cd "$HERE"
exec "$PY" watch.py --interval "${BENCHSUITE_INTERVAL_S:-300}"
