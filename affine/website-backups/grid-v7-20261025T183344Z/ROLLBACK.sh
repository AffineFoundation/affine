#!/usr/bin/env bash
# Roll back grid v7 (simplification pass) to grid v6
set -euo pipefail
cp -a /home/const/subnet120/affine/website-backups/grid-v7-20261025T183344Z/index.html /home/const/subnet120/affine/website-backups/grid-v7-20261025T183344Z/grid.js /home/const/subnet120/affine/website-backups/grid-v7-20261025T183344Z/grid.css /home/const/subnet120/affine/website-backups/grid-v7-20261025T183344Z/network.js /home/const/subnet120/affine/website-backups/grid-v7-20261025T183344Z/site-metrics.js /home/const/subnet120/affine/website-backups/grid-v7-20261025T183344Z/flip.js /home/const/subnet120/affine/website/
echo "rolled back affine.io to grid v6"
