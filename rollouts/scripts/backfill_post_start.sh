#!/bin/bash
# Env-backfill driver pod: /post_start.sh hook (PID 1 runs it on every
# container start). No datagen supervisor here — the coverage queue starts
# `rollouts.backfill` drivers over ssh (ops/coverage/start_env_backfill.sh),
# one tmux session per model. This hook only (re)starts the health endpoint
# (rollouts/scripts/backfill_health.py, port 20000) once docker is up, and
# re-attaches nothing: a driver that died with the container is restarted by
# the coverage queue's own release/retry logic.
mkdir -p /root/logs
if pgrep -f "backfill_health.py" >/dev/null 2>&1; then
  echo "[post_start] $(date -u) backfill health endpoint already running" >> /root/logs/backfill_health.log
  exit 0
fi
nohup bash -c '
  for i in $(seq 1 180); do docker info >/dev/null 2>&1 && break; sleep 5; done
  echo "[post_start] $(date -u) docker ready after ${i}x5s; starting backfill health endpoint"
  export BACKFILL_POD_NAME="$(cat /root/rollouts/.pod_name 2>/dev/null)"
  exec python3 /root/rollouts/scripts/backfill_health.py
' >> /root/logs/backfill_health.log 2>&1 < /dev/null &
disown
exit 0
