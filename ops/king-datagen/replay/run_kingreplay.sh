#!/bin/bash
# Current-king replay pass on an env-backfill box (coordinator "go" 2026-09-23 22:13):
# the datagen pods' teacher-solved tasks under the plain king_* ids into the LIVE
# traces/ prefix, one source after another. $1 = shard i/n (box split).
set -uo pipefail
cd /root/rollouts
source /root/affine/.datagen_env; source /root/rollouts/.rollouts_env
export PATH=/root/.local/bin:$PATH PYTHONPATH=/root/affine:/root/rollouts
export ROLLOUTS_R2_PREFIX=traces/ ROLLOUTS_KING_ENV=/root/rollouts/.king_env_replay
export ROLLOUTS_DATA_DIR=/root/rollouts-data-kingreplay ROLLOUTS_CATALOG_DIR=/root/rollouts-data/catalogs
export ROLLOUTS_MAX_CONTAINERS=72 ROLLOUTS_BATCH_SIZE=24 ROLLOUTS_MAX_LOCAL_BUILDS=${KINGREPLAY_MAX_BUILDS:-24}
# a king rollout still running after 20 min on these tasks is a loop (= failed for the band); do not let 2 stragglers hold 46 finished ones for an hour
export ROLLOUTS_ROLLOUT_TIMEOUT_S=1200 ROLLOUTS_BATCH_TIMEOUT_S=2400
SHARD=${1:-0/1}
for spec in "terminal_lego king_textbased:1" "scaleswe king_textbased:1,king_bashtool:1" "swesmith king_textbased:1"; do
  set -- $spec; src=$1; pols=""
  for p in ${2//,/ }; do pols="$pols --policy $p"; done
  echo "[kingreplay] $(date -u +%FT%TZ) start $src shard $SHARD ($pols)"
  /root/venv/bin/python -m rollouts.prepass --source $src $pols --uids-file /root/kingreplay/$src.uids --shard "$SHARD" --parallel 3 --worker-batch 24
  echo "[kingreplay] $(date -u +%FT%TZ) end $src rc=$?"
done
echo "[kingreplay] $(date -u +%FT%TZ) all done"
