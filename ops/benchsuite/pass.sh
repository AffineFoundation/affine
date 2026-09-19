#!/bin/bash
# One benchmark pass by hand, with the same env the pm2 watcher gets:
#   ops/benchsuite/pass.sh <ref> <label> <run_id> [MODE]   (env: CHALLENGER_REVISION etc. pass through)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="${BENCHSUITE_PYTHON:-$REPO/.venv/bin/python}"
# shellcheck disable=SC1091
source "$HERE/env.sh"
exec bash "$HERE/run_pass.sh" "$@"
