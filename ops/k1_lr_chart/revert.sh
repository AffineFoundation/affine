#!/usr/bin/env bash
# Revert the affine.io "Learning rate per turn" tile (Jacob pt1039u, 2026-11-01). Run on affinevalidator.
set -euo pipefail
TS=20261102T015639Z; W=$HOME/subnet120/affine/website; C=$HOME/subnet120-rewrite/state/root-audits/20261026-promotion-gate/code
for f in index.html k1.js tile-details.js; do cp -p $W/$f.bak-pre-lrchart-$TS $W/$f; done
cp -p $C/training_params.py.bak-pre-lrchart-$TS $C/training_params.py
systemctl --user restart affine-training-params
rm -f $W/k1/lr.json
echo reverted
