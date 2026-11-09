#!/usr/bin/env bash
# One-command PIT live revert (CUTOVER-PLAN step 6 revert).
# Restores trainer dequeue / warm / standing lanes via pe_live + pe_lifecycle.
# Stops affine-pit-host when the unit is present. Does not touch weight nonces.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
export PYTHONPATH="${PYTHONPATH:-}:${REPO_ROOT}"

PE_FLAG="${PE_FLAG:-}"
PE_STATE="${PE_STATE:-}"
PE_LIFE_STATE="${PE_LIFE_STATE:-}"
PIT_STATE="${PIT_STATE:-${PE_STATE:+$PE_STATE/pit}}"
CLAIMS_DIR="${CLAIMS_DIR:-${PIT_STATE:+$PIT_STATE/claims}}"
RESTORE_FLAG_BACKUP="${RESTORE_FLAG_BACKUP:-0}"

die() { echo "cutover_revert: $*" >&2; exit 2; }
[[ -n "$PE_FLAG" && -f "$PE_FLAG" ]] || die "set PE_FLAG to the live PROMOTION-GATE / PE flag JSON"

if [[ -z "$PE_STATE" ]]; then
  PE_STATE="$(dirname "$PE_FLAG")/state"
fi

LOCK="${CUT_OVER_LOCK:-}"
if [[ -z "$LOCK" ]]; then
  if [[ -n "$PE_STATE" ]]; then
    LOCK="$PE_STATE/cutover.lock"
  else
    LOCK="${PE_FLAG}.lock"
  fi
fi
mkdir -p "$(dirname "$LOCK")"
exec 9>"$LOCK"
if ! flock -x 9; then
  die "could not acquire cutover lock $LOCK"
fi

if systemctl --user is-active --quiet affine-pit-host.service 2>/dev/null; then
  systemctl --user stop affine-pit-host.service || true
  echo "cutover_revert: stopped affine-pit-host"
fi

export PE_STATE PE_LIFE_STATE PIT_STATE CLAIMS_DIR
python3 -m ops.parallel_eval.pit.cutover_ops clear-claims

python3 -m ops.parallel_eval.pit.cutover_ops release-pit-boxes || echo "cutover_revert: warning — pit box release incomplete (see above)" >&2

if [[ -n "$PE_STATE" ]]; then
  python3 - <<'PY' "$PE_STATE"
import sys
from pathlib import Path
from ops.parallel_eval.pit import cutover_ops
cutover_ops.remove_warm_off(Path(sys.argv[1]))
print("cutover_revert: removed WARM-OFF")
PY
fi

export PE_FLAG
if [[ "$RESTORE_FLAG_BACKUP" == "1" ]]; then
  BAK="$(python3 - <<'PY' "$PE_FLAG"
import sys
from pathlib import Path
from ops.parallel_eval.pit.cutover_ops import find_latest_pre_pit_backup
p = find_latest_pre_pit_backup(Path(sys.argv[1]))
print(p or "")
PY
)"
  if [[ -n "$BAK" && -f "$BAK" ]]; then
    export RESTORE_FROM="$BAK"
    echo "cutover_revert: restoring flag from $BAK"
    python3 -m ops.parallel_eval.pit.cutover_ops revert-patch
  else
    echo "cutover_revert: no .bak-pre-pit-* found; patching in place" >&2
    unset RESTORE_FROM
    cp -a "$PE_FLAG" "${PE_FLAG}.bak-pre-revert-$(date -u +%Y%m%dT%H%M%SZ)"
    python3 -m ops.parallel_eval.pit.cutover_ops revert-patch
  fi
else
  cp -a "$PE_FLAG" "${PE_FLAG}.bak-pre-revert-$(date -u +%Y%m%dT%H%M%SZ)" 2>/dev/null || true
  python3 -m ops.parallel_eval.pit.cutover_ops revert-patch
fi

echo "cutover_revert: pit.revert=true, enabled=false; pe_live.dispatch_tick / warm may resume"
echo "cutover_revert: re-rent standing pods with replace_pod.py if they were released"
