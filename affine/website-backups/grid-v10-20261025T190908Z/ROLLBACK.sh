#!/bin/sh
# Restore v9 versions of files changed in v10 (flip credit neutral bars)
set -e
W=/home/const/subnet120/affine/website; B=/home/const/subnet120/affine/website-backups/grid-v10-20261025T190908Z
for f in index.html grid.js flip.js; do cp $B/$f $W/.$f.new && mv $W/.$f.new $W/$f; done
echo rolled back to v9
