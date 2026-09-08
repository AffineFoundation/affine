#!/bin/bash
# Affine datagen: relaunch the rollouts supervisor after a container (re)start.
# /start.sh (PID 1) runs this file synchronously, so everything goes to the
# background; the docker daemon (DinD) comes up after PID 1, so wait for it.
# Installed 2026-09-07 after pods 2/3 stayed idle for 3 days following a
# host-side restart on 2026-09-04.
mkdir -p /root/logs
if pgrep -f "^bash /root/rollouts/bootstrap.sh" >/dev/null 2>&1; then
  echo "[post_start] $(date -u) rollouts bootstrap already running" >> /root/logs/bootstrap.log
  exit 0
fi
nohup bash -c '
  for i in $(seq 1 180); do docker info >/dev/null 2>&1 && break; sleep 5; done
  echo "[post_start] $(date -u) docker ready after ${i}x5s; launching rollouts bootstrap"
  cd /root/rollouts && exec bash /root/rollouts/bootstrap.sh
' >> /root/logs/bootstrap.log 2>&1 < /dev/null &
disown
exit 0
