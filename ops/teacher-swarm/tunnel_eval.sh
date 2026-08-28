#!/bin/bash
# Reverse tunnel: eval pod's 127.0.0.1:9100 -> swarm router here (:9100).
# The evalsrv teacher pool reaches the swarm through this — nothing public.
# Loop forever; ssh exits on link death and we redial. Run via swarmctl.
# The eval pod address is re-read from validator state each dial (2026-08-27:
# the provisioner replaces pods; a hardcoded default stranded the tunnel on a
# dead pod after the wvk-10 reset).
set -u
STATE_JSON=${STATE_JSON:-/home/const/subnet120/affine/state/state.json}
while true; do
  addr=$(python3 -c "
import json, sys
try:
    em = json.load(open('$STATE_JSON')).get('eval_machine') or {}
    ssh = em.get('ssh', '')          # 'root@HOST -p PORT'
    host = ssh.split('@')[1].split()[0]
    port = ssh.split('-p')[1].strip()
    print(host, port)
except Exception:
    sys.exit(1)
") || { echo "$(date -u +%FT%TZ) no eval machine in state; retry in 30s" >&2; sleep 30; continue; }
  EVAL_HOST=${addr% *}
  EVAL_PORT=${addr#* }
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
