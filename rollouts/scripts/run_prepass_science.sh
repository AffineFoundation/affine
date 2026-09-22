#!/bin/bash
# Teacher 3-attempt + king 2-attempt pre-pass over affine_science (aa-gap-fill plan item 1;
# fold [band_filter.affine_science] counts attempts from the traces). Shard given as $1 (i/n).
set -uo pipefail
cd /root/rollouts
source /root/affine/.datagen_env; source /root/rollouts/.rollouts_env
export PATH=/root/.local/bin:$PATH PYTHONPATH=/root/affine:/root/rollouts
export ROLLOUTS_DATA_DIR=/root/rollouts-data-prepass-science ROLLOUTS_MAX_CONTAINERS=24 ROLLOUTS_BATCH_SIZE=48
[ "${ROLLOUTS_R2_PREFIX:-traces/}" = traces/ ] || { echo "refusing: ROLLOUTS_R2_PREFIX=${ROLLOUTS_R2_PREFIX:-} (pre-pass must publish to traces/)"; exit 2; }
exec /root/venv/bin/python -m rollouts.prepass --source affine_science \
  --policy teacher_boxed:3 --policy king_boxed:2 --budget-usd 65 --shard "${1:-0/1}" --parallel 3 --worker-batch 48
