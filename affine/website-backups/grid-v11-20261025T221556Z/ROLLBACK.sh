#!/bin/sh
# Restore v10 versions of files changed in v11 (flip share table columns)
set -e
W=/home/const/subnet120/affine/website; B=/home/const/subnet120/affine/website-backups/grid-v11-20261025T221556Z
for f in index.html grid.js flip.js grid.css; do cp $B/$f $W/.$f.new && mv $W/.$f.new $W/$f; done
echo rolled back to v10
