#!/usr/bin/env bash
# Uncrown reign 13 -> reign 12. MUST run while affine-validator is stopped
# (deploy_wvk16.sh calls it at the duel boundary). Standalone use:
#   pm2 stop affine-validator && bash ops/incident-r13/revert_reign13.sh && pm2 start affine-validator
set -euo pipefail
REPO=/home/const/subnet120
cd "$REPO"
source .venv/bin/activate
ts() { date -u +%FT%TZ; }
if pm2 pid affine-validator 2>/dev/null | grep -qE '^[1-9][0-9]*$'; then
  echo "$(ts) affine-validator is RUNNING — stop it first (State.load must re-read the files); abort"; exit 1
fi
python ops/incident-r13/revert_reign13.py --check
python ops/incident-r13/revert_reign13.py --apply
echo "$(ts) kingctl view (reads state.json on its next 60 s tick; reign-13 box is removed as never-published):"
timeout 90 python ops/king-datagen/kingctl.py status 2>&1 | head -12 || true
