#!/bin/bash
# Revert the Queue count fix (Jacob, deployed 20261102T173805Z) to the first tables version. Full revert of the tables: ../20261102T173422Z/revert.sh
set -eu
W=/home/const/subnet120/affine/website; B=/home/const/subnet120-rewrite/state/bench-results/site-backup/20261102T173805Z
cp -p "$B/index.html" "$B/k1-tables.js" "$B/k1-tables.css" "$W/"
echo "restored $B"
