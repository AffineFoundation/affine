#!/usr/bin/env bash
# Roll back grid v5 (Performance removed, Epoch timings moved to row 2) to grid v4
set -euo pipefail
cp -a /home/const/subnet120/affine/website-backups/grid-v5-20261025T182131Z/index.html /home/const/subnet120/affine/website-backups/grid-v5-20261025T182131Z/grid.js /home/const/subnet120/affine/website-backups/grid-v5-20261025T182131Z/grid.css /home/const/subnet120/affine/website/
echo "rolled back affine.io to grid v4"
