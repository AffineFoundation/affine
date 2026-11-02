#!/bin/sh
# Revert pt1076u speed fixes (slices, 200 ms fade, zstd, edge cache headers) back to the pt1070u state backed up at 20261102T155908Z.
# For a full revert to before both changes, run this and then site-backup/20261102T154146Z/revert.sh.
set -e
B=/home/const/subnet120-rewrite/state/bench-results/site-backup/20261102T155908Z
W=/home/const/subnet120/affine/website
for f in index.html baseline/index.html load.js load.css timing.js top.js regcost.js; do cp -p $B/website/$f $W/$f; done
systemctl --user disable --now affine-site-slices.timer 2>/dev/null || true
rm -f ~/.config/systemd/user/affine-site-slices.service ~/.config/systemd/user/affine-site-slices.timer; systemctl --user daemon-reload
rm -f $W/site-metrics-recent.json $W/gate-recent.json $W/reg-cost-recent.json /home/const/subnet120/affine/site-tools/build_site_slices.py
if ! cmp -s $B/Caddyfile /etc/caddy/Caddyfile; then sudo cp $B/Caddyfile /etc/caddy/Caddyfile && sudo caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile && sudo systemctl reload caddy; fi
echo reverted pt1076u speed fixes to 20261102T155908Z
