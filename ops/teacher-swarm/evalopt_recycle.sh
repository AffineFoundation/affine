#!/bin/bash
# Recycle eval-box replicas with new env knobs. Run FROM the pod as root.
# Usage: GPU_UTIL=0.75 BATCHED_TOKENS=8192 EXTRA_VLLM_ARGS='...' bash evalopt_recycle.sh
set -euo pipefail
ENV=/root/swarm/env
[ -f "$ENV" ] || { echo "no $ENV"; exit 2; }
set -a
source "$ENV"
set +a
if [ -n "${GPU_UTIL_OVERRIDE:-}" ]; then
  sed -i "s/^GPU_UTIL=.*/GPU_UTIL=\"$GPU_UTIL_OVERRIDE\"/" "$ENV"
fi
if [ -n "${BATCHED_OVERRIDE:-}" ]; then
  sed -i "s/^BATCHED_TOKENS=.*/BATCHED_TOKENS=\"$BATCHED_OVERRIDE\"/" "$ENV"
fi
if [ -n "${EXTRA_OVERRIDE:-}" ]; then
  # single-quoted value; EXTRA_OVERRIDE must not contain single quotes
  sed -i "s|^EXTRA_VLLM_ARGS=.*|EXTRA_VLLM_ARGS='$EXTRA_OVERRIDE'|" "$ENV"
fi
set -a
source "$ENV"
set +a
echo "recycling util=$GPU_UTIL chunk=$BATCHED_TOKENS extra=${EXTRA_VLLM_ARGS:-}"
# Stop supervisor + engines; manager will see them dark and re-bootstrap
# unless we restart supervisor ourselves.
if [ -f /root/swarm/boot.pid ]; then
  kill "$(cat /root/swarm/boot.pid)" 2>/dev/null || true
fi
pkill -f "vllm serve" || true
sleep 3
setsid nohup bash /root/swarm/bootstrap.sh >> /root/swarm/bootstrap.log 2>&1 < /dev/null &
echo $! > /root/swarm/boot.pid
echo "supervisor restarted pid=$!"
