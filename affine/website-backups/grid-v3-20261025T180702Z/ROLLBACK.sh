#!/usr/bin/env bash
# Roll back grid v3 (probe + miners/batches/groups reorder, clean rollouts chart) to grid v2
set -euo pipefail
cp -a /home/const/subnet120/affine/website-backups/grid-v3-20261025T180702Z/index.html /home/const/subnet120/affine/website-backups/grid-v3-20261025T180702Z/grid.js /home/const/subnet120/affine/website-backups/grid-v3-20261025T180702Z/site-metrics.js /home/const/subnet120/affine/website/
echo "rolled back affine.io to grid v2"
