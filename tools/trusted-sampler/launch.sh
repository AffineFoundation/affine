#!/usr/bin/env bash
# Launch N independent sampler instances, one per GPU, each with its own random K.
# usage: ./launch.sh [N=8] [extra sampler.py args...]     (default: --watch -K 256 -m 8)
#   e.g. ./launch.sh 8 --watch -K 256 -m 8
#        ./launch.sh 4 --backfill 6 -K 128 -m 8
#        GPU_OFFSET=4 ./launch.sh 4 --pair cf791667:19f21658 -K 256 -m 8
set -euo pipefail
cd "$(dirname "$0")"
N=${1:-8}; shift || true
ARGS=("$@"); [ ${#ARGS[@]} -eq 0 ] && ARGS=(--watch -K 256 -m 8)
PY=${PY:-/root/sampler-venv/bin/python}; OFF=${GPU_OFFSET:-0}
mkdir -p logs results
for i in $(seq 0 $((N-1))); do
  g=$((OFF+i)); id="$(hostname)-g${g}-$(head -c4 /dev/urandom | od -An -tx1 | tr -d ' \n')"
  CUDA_VISIBLE_DEVICES=$g setsid nohup "$PY" sampler.py --sampler-id "$id" "${ARGS[@]}" \
      > "logs/${id}.log" 2>&1 < /dev/null &
  echo "started $id on GPU $g (pid $!)"
  # stagger engine start-up: concurrent first-time torch.compile cache writes can race
  if [ "$i" -lt $((N-1)) ]; then sleep "${STAGGER:-20}"; fi
done
