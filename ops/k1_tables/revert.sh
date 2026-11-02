#!/bin/bash
# Revert the Kings / Queue / History tables (Jacob pt1100u, deployed 20261102T173422Z): restore index.html + k1.js, remove the new
# files and the affine-site-k1-tables timer. Safe to re-run.
set -u
W=/home/const/subnet120/affine/website
B=/home/const/subnet120-rewrite/state/bench-results/site-backup/20261102T173422Z
cp -p "$B/index.html" "$W/index.html"
cp -p "$B/k1.js" "$W/k1.js"
rm -f "$W/k1-tables.js" "$W/k1-tables.css" "$W/k1/tables.json"
systemctl --user disable --now affine-site-k1-tables.timer 2>/dev/null || true
rm -f /home/const/.config/systemd/user/affine-site-k1-tables.timer /home/const/.config/systemd/user/affine-site-k1-tables.service
systemctl --user daemon-reload
rm -f /home/const/subnet120/affine/site-tools/build_k1_tables.py
echo "reverted to the $B snapshot"
