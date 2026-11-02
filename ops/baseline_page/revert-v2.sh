#!/bin/sh
# Revert affine.io/baseline v2 ('How the draw works' section + caption fix, Jacob pt1059u). Restores baseline/index.html, baseline.css, baseline.js and site-tools/build_baseline_curves.py as of 20261102T151156Z (= commit d94522e state).
set -e
cp -p /home/const/subnet120-rewrite/state/bench-results/site-backup/20261102T151156Z/build_baseline_curves.py /home/const/subnet120/affine/site-tools/
cp -p /home/const/subnet120-rewrite/state/bench-results/site-backup/20261102T151156Z/index.html /home/const/subnet120-rewrite/state/bench-results/site-backup/20261102T151156Z/baseline.css /home/const/subnet120-rewrite/state/bench-results/site-backup/20261102T151156Z/baseline.js /home/const/subnet120/affine/website/baseline/
echo reverted affine.io/baseline to 20261102T151156Z
