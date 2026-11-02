#!/bin/sh
# Revert affine.io/baseline IRT overlay (Jacob pt1080u): restores baseline/index.html, baseline.css, baseline.js as of 20261102T161642Z (= commit e808c0a state)
# and removes the static IRT overlay data + its one-off builder. The private fit (RA/20261102-irt-baseline) is left in place.
set -e
B=/home/const/subnet120-rewrite/state/bench-results/site-backup/20261102T161642Z
W=/home/const/subnet120/affine/website
cp -p $B/website/baseline/index.html $B/website/baseline/baseline.css $B/website/baseline/baseline.js $W/baseline/
rm -f $W/baseline/irt-curves.json /home/const/subnet120/affine/site-tools/build_baseline_irt_curves.py
echo reverted affine.io/baseline IRT overlay to 20261102T161642Z
