#!/bin/bash
# Pod-side launcher for the coached-teacher recovery run. Layout on the pod:
#   /root/coached/code/{recoverable,hints}   ops/recoverable + research/hints (this dir inside)
#   /root/coached/run/<RUN>/states           states.jsonl + states/ + coach_ctx/ (select_states.py)
#   /root/coached/run/<RUN>/out              results, traces, coach logs
#   /root/coached/.env                        ENGY_2 (teacher key), OPENROUTER_API_KEY (coach)
# Reads the pod's rollouts env read-only (source -> eval flag mapping); the
# datagen supervisor is never started here.
#
#   pod_run.sh proxy <RUN> [proxy args...]     start the coach proxy (foreground)
#   pod_run.sh run   <RUN> [run args...]       run continuations through it
set -euo pipefail
MODE="$1"; RUN="$2"; shift 2
BASE=/root/coached
set -a
# shellcheck disable=SC1091
source /root/affine/.datagen_env
# shellcheck disable=SC1091
source /root/rollouts/.rollouts_env
# shellcheck disable=SC1091
source "$BASE/.env"
set +a
export PYTHONPATH=/root/affine:/root/rollouts
export RECOVERABLE_SRC="$BASE/code/recoverable"
export RECOVERABLE_PLUGIN="$BASE/code/recoverable/plugin"
export PATH="$HOME/.local/bin:$PATH"
PY=/root/venv/bin/python
CODE="$BASE/code/hints/coached"
RUN_DIR="$BASE/run/$RUN"
case "$MODE" in
  proxy)
    exec "$PY" "$CODE/coach_proxy.py" --ctx-dir "$RUN_DIR/states/coach_ctx" \
      --log-dir "$RUN_DIR/out/coach" "$@" ;;
  run)
    exec "$PY" "$CODE/run_coached.py" --states "$RUN_DIR/states/states.jsonl" \
      --out "$RUN_DIR/out" "$@" ;;
  *) echo "usage: $0 proxy|run <RUN> [args]" >&2; exit 2 ;;
esac
