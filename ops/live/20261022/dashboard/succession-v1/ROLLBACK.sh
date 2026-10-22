#!/bin/bash
# Restore the pre-20261017 dashboard drop-in and restart only the dashboard unit.
set -euo pipefail
U=/home/const/.config/systemd/user/affine-network-dashboard.service.d
cp -a /home/const/subnet120-rewrite/state/root-audits/20261017-dashboard-succession-v1/service-override.ROLLBACK.conf "$U/zzzz~mimo-data-projection.conf"
systemctl --user daemon-reload
systemctl --user restart affine-network-dashboard.service
systemctl --user --no-pager status affine-network-dashboard.service | head -5
