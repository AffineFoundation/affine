#!/bin/bash
# Run the teacher-continuation driver on a datagen pod, in its own working
# directory (/root/recoverable), reading the pod's env files read-only.
#   /root/recoverable/recoverable/pod_run.sh --states ... --out ... --kinds ... --workers N
set -euo pipefail
set -a
# shellcheck disable=SC1091
source /root/affine/.datagen_env
# shellcheck disable=SC1091
source /root/rollouts/.rollouts_env
# shellcheck disable=SC1091
source /root/recoverable/.env
set +a
export PYTHONPATH=/root/affine:/root/rollouts
export RECOVERABLE_PLUGIN=/root/recoverable/recoverable/plugin
export PATH="$HOME/.local/bin:$PATH"
exec /root/venv/bin/python /root/recoverable/recoverable/run_states.py "$@"
