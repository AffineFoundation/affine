#!/bin/bash
# Reverse tunnel: eval pod's 127.0.0.1:9100 -> swarm router here (:9100).
# The evalsrv teacher pool reaches the swarm through this — nothing public.
# Loop forever; ssh exits on link death and we redial. Run via swarmctl.
set -u
EVAL_HOST=${EVAL_HOST:-152.236.142.234}
EVAL_PORT=${EVAL_PORT:-40300}
while true; do
  ssh -N \
    -o StrictHostKeyChecking=accept-new \
    -o UserKnownHostsFile=state/known_hosts \
    -o ConnectTimeout=10 \
    -o ServerAliveInterval=15 \
    -o ServerAliveCountMax=3 \
    -o ExitOnForwardFailure=yes \
    -R 127.0.0.1:9100:127.0.0.1:9100 \
    -p "$EVAL_PORT" "root@$EVAL_HOST"
  echo "$(date -u +%FT%TZ) tunnel dropped (exit $?), redialing in 5s" >&2
  sleep 5
done
