#!/bin/bash
# Frontier-arbiter outcome probe: run the continuation driver on a datagen pod
# from its OWN working directory (/root/recoverable/frontier), reading the
# pod's env files read-only. Same shape as ops/recoverable/pod_run.sh.
#   /root/recoverable/frontier/recoverable/pod_run.sh --states ... --out ... --kinds ... --workers N [--model glm-5.3]
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
export RECOVERABLE_PLUGIN=/root/recoverable/frontier/recoverable/plugin
export PATH="$HOME/.local/bin:$PATH"
exec /root/venv/bin/python /root/recoverable/frontier/recoverable/run_states.py "$@"
