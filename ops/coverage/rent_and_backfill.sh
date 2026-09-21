#!/bin/bash
# Rent one serving box for a model (cheapest in-stock plan first), record it in
# the pod ledger, then run start_env_backfill.sh (driver on the backfill pod,
# release when done). Called by autofill.py; usable by hand:
#   rent_and_backfill.sh <ref: sha256 | hf://repo@rev> <digest12> <label> "<sources|all>" [containers]
set -uo pipefail
cd ~/subnet120/ops/benchsuite
HERE=$PWD; REPO=~/subnet120; PY=$REPO/.venv/bin/python
source ./env.sh
export LIUM_API_KEY="${LIUM_API_KEY:-${LIUM:-}}"
REF="$1"; D12="$2"; LABEL="$3"; SOURCES="$4"; MAXC="${5:-12}"
LEDGER=~/subnet120/ops/coverage/state/backfill_pods.json
LOG=~/subnet120/ops/coverage/state/env_backfill_$D12.log
log() { echo "[rent_and_backfill $LABEL] $(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
if [[ "$REF" == hf://* ]]; then DIGEST="${REF##*@}"; EXTRA=(--hf "${REF#hf://}"); else DIGEST="$REF"; EXTRA=(); fi
POD=""
for attempt in $(seq 1 8); do
  for PLAN in pro6000-1x h200-1x b200-1x h100-2x; do
    POD=$($PY kingpod.py rent --plan "$PLAN" --digest "$DIGEST" "${EXTRA[@]}" 2>>"$LOG" | tail -1) && [ -n "$POD" ] && break
    POD=""
  done
  [ -n "$POD" ] && break
  log "no stock on any plan (attempt $attempt); retry in 15 min"; sleep 900
done
[ -n "$POD" ] || { log "giving up: no stock"; exit 2; }
log "rented $POD for $LABEL ($D12)"
$PY - "$LEDGER" "$POD" "$LABEL" "$D12" <<'PY'
import json, sys, time
from pathlib import Path
led, pod, label, d12 = sys.argv[1:]
p = Path(led); rows = json.loads(p.read_text()) if p.exists() else []
mem = json.load(open("state/pods.json"))[pod]
rows.append({"pod": pod, "owner": "coverage", "purpose": "model", "label": label, "model": d12,
             "plan": mem["plan"]["name"], "machine": mem.get("machine"), "usd_per_hour": mem["price"],
             "rented_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(mem["rented_at"])), "released_at": None})
p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(rows, indent=1))
PY
exec bash ~/subnet120/ops/coverage/start_env_backfill.sh "$POD" "$D12" "$LABEL" "$SOURCES" "$MAXC"
