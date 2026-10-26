#!/bin/sh
# Restore v8 (v7 grid + bans section) versions of the files changed in v9
set -e
W=/home/const/subnet120/affine/website; B=/home/const/subnet120/affine/website-backups/grid-v9-20261025T190333Z
for f in index.html grid.js grid.css site-metrics.js flip.js; do cp $B/$f $W/.$f.new && mv $W/.$f.new $W/$f; done
echo rolled back to v8
