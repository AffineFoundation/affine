#!/bin/bash
# Env-backfill driver entry point on the driver pod (installed as
# /root/rollouts/run_backfill.sh by clone_datagen_pod.sh CLONE_ROLE=backfill).
# The coverage queue's start_env_backfill.sh writes a per-model copy
# (run_backfill_<digest12>.sh) with its own data dir / king env; this generic
# one is the by-hand form and the file the queue's reachability check looks
# for. Refuses to run unless traces go to the backfill prefix: a driver that
# published into `traces/` would feed the live corpus D.
#   run_backfill.sh --digest12 <d12> --n 50 [--sources 'a b'|all] [--dry-run]
set -uo pipefail
cd /root/rollouts
source /root/affine/.datagen_env; source /root/rollouts/.rollouts_env
export PATH=/root/.local/bin:$PATH PYTHONPATH=/root/affine:/root/rollouts
[ "${ROLLOUTS_R2_PREFIX:-}" = traces-backfill/ ] || { echo "refusing: ROLLOUTS_R2_PREFIX=${ROLLOUTS_R2_PREFIX:-} (must be traces-backfill/)"; exit 2; }
[ -e /root/rollouts/.king_env ] && { echo "refusing: /root/rollouts/.king_env exists (live king seat on a backfill pod)"; exit 2; }
exec /root/venv/bin/python -m rollouts.backfill "$@"
