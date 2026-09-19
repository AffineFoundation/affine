#!/bin/bash
# pm2 wrapper for affine-pod-reaper.
#   pm2 start ops/pods/run_reaper.sh --name affine-pod-reaper --interpreter bash -- --interval 900
#   pm2 start ops/pods/run_reaper.sh --name affine-pod-reaper --interpreter bash -- --dry-run
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO/ops/health/pm2_env.sh"
cd "$REPO/ops/pods"
exec "$REPO/.venv/bin/python" reaper.py "$@"
