#!/usr/bin/env bash
# Revert the pass-rate chart fix (pt904u 2026-10-31): restore the pre-fix builder and site files, restart the K=1 service.
set -euo pipefail
D=/home/const/subnet120-rewrite/state/root-audits/20261031-pass-chart-fix/backup
C=/home/const/subnet120-rewrite/state/root-audits/20261027-k1-per-uid/code
W=/home/const/subnet120/affine/website
install -m 644 "$D/k1_service.py" "$C/k1_service.py.new" && mv "$C/k1_service.py.new" "$C/k1_service.py"
install -m 644 "$D/k1.js" "$W/k1.js.new" && mv "$W/k1.js.new" "$W/k1.js"
# index.html: only put the old k1.js cache-buster back (other jobs may have edited the page since)
sed -i 's#/k1.js?v=20261031-passrates#/k1.js?v=20261030-freefile#' "$W/index.html"
systemctl --user restart affine-k1
echo "reverted; k1.json is rewritten by affine-k1 within ~60 s (old format: no pass rates from E272)"
