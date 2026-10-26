# Rollback: affine.io "Rollouts per epoch" chart (deployed 2026-10-24 19:24 UTC = 13:24 CR)
Backup on ArbosLife: /home/const/subnet120/affine/website-backups/rollouts-chart-20261024T192429Z/
  (site-metrics.js, site-metrics.css, site-metrics.json, site_metrics.py, cache.json — originals)
Restore (on ArbosLife as const):
  B=/home/const/subnet120/affine/website-backups/rollouts-chart-20261024T192429Z
  cp -p $B/site-metrics.js $B/site-metrics.css /home/const/subnet120/affine/website/
  cp -p $B/site_metrics.py /home/const/subnet120-rewrite/dashboard/site_metrics.py
  systemctl --user start affine-site-metrics.service   # regenerates site-metrics.json (v1; cache rebuilds automatically)
Front-end only rollback (keep richer JSON; it is a superset): restore just the two website files.
Local copies of originals: /workspace/affine-rollouts-chart/{site-metrics.orig.js,site-metrics.orig.css,site_metrics.orig.py}
