#!/bin/bash
# pm2 wrapper for affine-pipeline-health.
#   pm2 start ops/health/run_health.sh --name affine-pipeline-health --interpreter bash -- --interval 300
#   pm2 start ops/health/run_health.sh --name affine-pipeline-health --interpreter bash -- --dry-run
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO/ops/health/pm2_env.sh"
cd "$REPO/ops/health"
exec "$REPO/.venv/bin/python" health.py "$@"
