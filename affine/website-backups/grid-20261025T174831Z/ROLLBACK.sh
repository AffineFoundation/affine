#!/usr/bin/env bash
# Roll back the 2026-10-25 uniform-grid redesign of affine.io
set -euo pipefail
B=/home/const/subnet120/affine/website-backups/grid-20261025T174831Z
W=/home/const/subnet120/affine/website
cp -a $B/index.html $B/site-metrics.js $B/flip.js $W/
rm -f $W/grid.css $W/grid.js
echo "rolled back affine.io to pre-grid files"
