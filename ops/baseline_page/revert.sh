#!/bin/sh
# Revert affine.io/baseline (Jacob 2026-11-02): restores index.html (nav link) and /etc/caddy/Caddyfile as of 20261102T142823Z,
# removes website/baseline/, the generator and the affine-site-baseline user timer, reloads Caddy (affine.io block only).
set -e
B=/home/const/subnet120-rewrite/state/bench-results/site-backup/20261102T142823Z
W=/home/const/subnet120/affine/website
export XDG_RUNTIME_DIR=/run/user/$(id -u)
systemctl --user disable --now affine-site-baseline.timer 2>/dev/null || true
rm -f $HOME/.config/systemd/user/affine-site-baseline.timer $HOME/.config/systemd/user/affine-site-baseline.service
systemctl --user daemon-reload || true
cp -p $B/index.html $W/index.html
rm -rf $W/baseline
rm -f $HOME/subnet120/affine/site-tools/build_baseline_curves.py
sudo cp $B/Caddyfile /etc/caddy/Caddyfile
sudo caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile >/dev/null 2>&1 && sudo systemctl reload caddy
echo reverted affine.io/baseline to 20261102T142823Z
