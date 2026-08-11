#!/bin/bash
# Unified rollouts bootstrap (datagen pod). Idempotent; ends by supervising
# the supervisor loop in a restart loop — same pattern as the loops it
# replaces (/root/affine/datagen/bootstrap.sh, /root/prime-lane/bootstrap.sh).
set -euo pipefail

cd /root/rollouts
export PATH="$HOME/.local/bin:$PATH"
mkdir -p /root/logs /root/rollouts-data

echo "[rollouts bootstrap] $(date -u) starting"

if ! docker info >/dev/null 2>&1; then
  echo "[rollouts bootstrap] FATAL: docker daemon is not running"; exit 1
fi
if [ ! -d "${ROLLOUTS_VERIFIERS_DIR:-/root/prime-pilot/verifiers}" ]; then
  echo "[rollouts bootstrap] FATAL: verifiers checkout missing"; exit 1
fi

# Import smoke: package + registry + production slicer/uploader.
source /root/affine/.datagen_env
if [ -f /root/rollouts/.rollouts_env ]; then
  # shellcheck disable=SC1091
  source /root/rollouts/.rollouts_env
fi
PYTHONPATH=/root/affine:/root/rollouts /root/venv/bin/python - <<'PY'
from rollouts.registry import load_registry
from rollouts.config import load_config
from datagen.slicer import slice_messages
from datagen.uploader import TurnUploader
r = load_registry()
cfg = load_config()
print(f"[rollouts bootstrap] IMPORT_OK sources={sorted(r.sources)} "
      f"policies={sorted(r.policies)} data_dir={cfg.data_dir}")
PY

# Supervise. `|| code=$?` is load-bearing: under `set -e` a bare nonzero
# exit of the loop would abort this script and the restart loop never runs.
while true; do
  # Re-source per launch so operators can tune ROLLOUTS_* and bounce the loop.
  source /root/affine/.datagen_env
  if [ -f /root/rollouts/.rollouts_env ]; then
    # shellcheck disable=SC1091
    source /root/rollouts/.rollouts_env
  fi
  echo "[rollouts bootstrap] $(date -u) launching rollouts supervisor"
  code=0
  PYTHONPATH=/root/affine:/root/rollouts /root/venv/bin/python \
    -m rollouts.run >> /root/logs/rollouts.log 2>&1 || code=$?
  echo "[rollouts bootstrap] $(date -u) supervisor exited code=$code; restarting in 30s"
  sleep 30
done
