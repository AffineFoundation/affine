#!/usr/bin/env bash
# Roll back grid v4 (title hover explanations) to grid v3
set -euo pipefail
cp -a /home/const/subnet120/affine/website-backups/grid-v4-20261025T181409Z/index.html /home/const/subnet120/affine/website-backups/grid-v4-20261025T181409Z/grid.js /home/const/subnet120/affine/website-backups/grid-v4-20261025T181409Z/grid.css /home/const/subnet120/affine/website/
echo "rolled back affine.io to grid v3"
