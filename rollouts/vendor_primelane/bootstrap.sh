#!/bin/bash
# Prime-lane bootstrap (second datagen lane on the datagen pod).
# Idempotent; ends by supervising the lane loop in a restart loop —
# same pattern as /root/affine/datagen/bootstrap.sh.
set -euo pipefail

cd /root/prime-lane
export PATH="$HOME/.local/bin:$PATH"
mkdir -p /root/logs /root/prime-lane/data

echo "[prime-lane bootstrap] $(date -u) starting"

if ! docker info >/dev/null 2>&1; then
  echo "[prime-lane bootstrap] FATAL: docker daemon is not running"; exit 1
fi
if [ ! -d /root/prime-pilot/verifiers ]; then
  echo "[prime-lane bootstrap] FATAL: verifiers checkout missing"; exit 1
fi

# Import smoke: lane package + production slicer/uploader + taskset ids.
source /root/affine/.datagen_env
if [ -f /root/prime-lane/.lane_env ]; then
  # shellcheck disable=SC1091
  source /root/prime-lane/.lane_env
fi
PYTHONPATH=/root/affine:/root/prime-lane /root/venv/bin/python - <<'PY'
from primelane.lanespec import SPECS, panel_keys
from primelane.lane import load_lane_config
from datagen.slicer import slice_messages
from datagen.uploader import TurnUploader
cfg = load_lane_config()
ids, repos, bare = panel_keys()
print(f"[prime-lane bootstrap] IMPORT_OK tasksets={ [n for n, _ in cfg.tasksets] } "
      f"providers={[p['name'] for p in cfg.providers]} panel_repos={len(repos)}")
PY

# Supervise. `|| code=$?` is load-bearing: under `set -e` a bare nonzero
# exit of the loop would abort this script and the restart loop never runs.
while true; do
  # Kill only THIS lane's orphans: eval processes run out of the verifiers
  # checkout; the main loop's mini-extra/swebench processes never match.
  pkill -f '[p]rime-pilot/verifiers' 2>/dev/null || true
  # Re-source per launch so operators can tune LANE_* and bounce the loop.
  source /root/affine/.datagen_env
  if [ -f /root/prime-lane/.lane_env ]; then
    # shellcheck disable=SC1091
    source /root/prime-lane/.lane_env
  fi
  echo "[prime-lane bootstrap] $(date -u) launching prime lane"
  code=0
  PYTHONPATH=/root/affine:/root/prime-lane /root/venv/bin/python \
    -m primelane.lane >> /root/logs/prime_lane.log 2>&1 || code=$?
  echo "[prime-lane bootstrap] $(date -u) lane exited code=$code; restarting in 30s"
  sleep 30
done
