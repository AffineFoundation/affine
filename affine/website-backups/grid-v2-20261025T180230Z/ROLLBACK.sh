#!/usr/bin/env bash
# Roll back the 2026-10-25 grid v2 change (incentive full-width bars, weights + mining tiles removed) to grid v1
set -euo pipefail
B=/home/const/subnet120/affine/website-backups/grid-v2-20261025T180230Z
W=/home/const/subnet120/affine/website
cp -a $B/index.html $B/network.js $B/grid.css $B/grid.js $B/site-metrics.js $B/flip.js $W/
echo "rolled back affine.io to grid v1"
