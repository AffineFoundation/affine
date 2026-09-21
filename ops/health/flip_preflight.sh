#!/usr/bin/env bash
# Step 0 of EVERY contract flip script (ops/vNN/deploy_wvkNN.sh pattern):
# nobody downstream may misread the numbers the new score_mode produces.
#
#   bash ops/health/flip_preflight.sh --score-mode <new mode> --wvk <new wvk>
#
# exit 0 = go: every consumer in ops/health/consumers.toml is compatible, or
#          was frozen (curriculum -> FROZEN.json, fold falls back to the
#          static [mix]) and the private channel was paged;
# exit 2 = BLOCKED: a consumer with on_mismatch = "block" (payout) does not
#          declare the new units. Re-declare it first; do not flip.
#
# The health monitor re-runs the same check every 5 min after the flip, so a
# flip that skips this step is caught within one tick — but a flip script
# that does not call this is a bug. Wire it right after `set -euo pipefail`:
#
#   bash "$REPO/ops/health/flip_preflight.sh" --score-mode sd_min_rga --wvk 23 \
#       || { echo "$(ts) contract preflight BLOCKED; abort"; exit 1; }
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$REPO/.venv/bin/python"
cd "$REPO"
echo "=== contract preflight $(date -u +%FT%TZ) $*"
"$PY" ops/health/contract_compat.py --preflight --page "$@"
rc=$?
if pm2 jlist 2>/dev/null | "$PY" -c 'import json,sys; d=json.load(sys.stdin); ok=any(p.get("name")=="affine-pipeline-health" and p["pm2_env"].get("status")=="online" for p in d); sys.exit(0 if ok else 1)'; then
  echo "affine-pipeline-health online"
else
  echo "WARNING: pm2 affine-pipeline-health is not online — start it before the flip (pm2 start ops/health/run_health.sh --name affine-pipeline-health --interpreter bash -- --interval 300)"
fi
if [[ $rc -eq 2 ]]; then
  echo "contract preflight: BLOCKED (a block-level consumer does not declare the new units)"
elif [[ $rc -eq 0 ]]; then
  echo "contract preflight: ok"
else
  echo "contract preflight: contract_compat exited $rc"
fi
exit $rc
