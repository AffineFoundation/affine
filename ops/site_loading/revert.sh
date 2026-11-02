#!/bin/sh
# Revert the chart load sequence (Jacob pt1070u, 20261102T154146Z): restores the 10 site files and the Caddyfile as of 20261102T154146Z,
# removes load.js/load.css, validates and gracefully reloads Caddy.
set -e
B=/home/const/subnet120-rewrite/state/bench-results/site-backup/20261102T154146Z
W=/home/const/subnet120/affine/website
for f in index.html perf.js bench.js k1.js top.js timing.js regcost.js network-desktop.js baseline/index.html baseline/baseline.js; do cp -p $B/website/$f $W/$f; done
rm -f $W/load.js $W/load.css
if ! cmp -s $B/Caddyfile /etc/caddy/Caddyfile; then sudo cp $B/Caddyfile /etc/caddy/Caddyfile && sudo caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile && sudo systemctl reload caddy; fi
echo reverted chart load sequence to 20261102T154146Z
