#!/usr/bin/env bash
# Roll back grid v6 (single-series Batches per epoch) to grid v5
set -euo pipefail
cp -a /home/const/subnet120/affine/website-backups/grid-v6-20261025T182420Z/index.html /home/const/subnet120/affine/website-backups/grid-v6-20261025T182420Z/grid.js /home/const/subnet120/affine/website-backups/grid-v6-20261025T182420Z/grid.css /home/const/subnet120/affine/website-backups/grid-v6-20261025T182420Z/network.js /home/const/subnet120/affine/website/
echo "rolled back affine.io to grid v5"
