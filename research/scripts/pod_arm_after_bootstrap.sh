#!/bin/bash
# Wait for bootstrap ALL_DONE then run a follow-up script.
# Usage: bash pod_arm_after_bootstrap.sh /root/logs/bootstrap_e6.log /root/pod_serve_wave5a.sh
set -euo pipefail
LOG=${1:?bootstrap log}
NEXT=${2:?next script}
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
if [ -f /root/.env.hf ]; then source /root/.env.hf; fi
export PYTHONPATH=/root
for i in $(seq 1 720); do
  if grep -q "ALL_DONE" "$LOG" 2>/dev/null; then
    echo "BOOTSTRAP_DONE at iter $i"
    bash "$NEXT"
    exit 0
  fi
  if grep -qiE "error|traceback|failed" "$LOG" 2>/dev/null && ! pgrep -f "snapshot_download|pod_bootstrap" >/dev/null; then
    echo "BOOTSTRAP_MAYBE_FAILED"; tail -40 "$LOG"; exit 1
  fi
  sleep 30
done
echo "TIMEOUT waiting for bootstrap"; tail -40 "$LOG"; exit 1
