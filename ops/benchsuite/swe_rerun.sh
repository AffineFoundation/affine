#!/bin/bash
# Re-run a card's SWE-bench Verified cell on Daytona at a fair in-flight (64 per replica),
# replacing a contended cell (200 agents on one replica: reign 18 = 454/500 timeouts).
#   bash swe_rerun.sh <digest> <label> <run_id>
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"; PY="${BENCHSUITE_PYTHON:-$REPO/.venv/bin/python}"; BENCH_HOME="${BENCH_HOME:-$HOME/benchsuite}"
# shellcheck disable=SC1091
source "$HERE/env.sh"
export LIUM_API_KEY="${LIUM_API_KEY:-${LIUM:-}}"; export HARBOR_BIN="${HARBOR_BIN:-$BENCH_HOME/harborenv/bin/harbor}"
[ -n "${DAYTONA_API_KEY:-}" ] || export DAYTONA_API_KEY=$(op read --no-newline "op://Arbos/fywmj6vtq5delybw5c7a53l2qa/notesPlain" 2>/dev/null | grep -o 'dtn_[A-Za-z0-9_-]*' | head -1)
DIGEST="$1" LABEL="$2" INTO="$BENCH_HOME/runs/$3"
log() { echo "[swe-rerun] $(date -u +%FT%TZ) $*"; }
POD=""; T0=$(date +%s)
while [ $(( $(date +%s) - T0 )) -lt "${SWE_RENT_DEADLINE_S:-14400}" ]; do   # zero Lium stock is normal tonight: wait up to 4 h
  for PLAN in ${SWE_PLANS:-h200-2x b200-2x h200-1x b200-1x pro6000-1x}; do POD=$("$PY" "$HERE/kingpod.py" rent --plan "$PLAN" --digest "$DIGEST" 2>/dev/null | tail -1) && [ -n "$POD" ] && break; POD=""; done
  [ -n "$POD" ] && break; log "$LABEL: no stock; retry in 3 min"; sleep 180
done
[ -n "$POD" ] || { log "$LABEL: no stock within the rent deadline"; exit 2; }
trap '"$PY" "$HERE/kingpod.py" release "$POD" >/dev/null 2>&1' EXIT
# a @4h250 job runs ~27 h at 64 in flight: tell the pod reaper (kingpod registers 14 h) so it does not release
# the serving box mid-job (2026-09-21: 19 / 16 / 15 lost 267-300 trials each to NetworkConnectionError that way)
"$PY" -c "import sys; sys.path.insert(0, '$REPO/ops/pods'); import registry; registry.register('$POD', expected_hours=36, source='explicit', meta={'job': 'swebench @4h250 / rerun'})" 2>/dev/null || true
"$PY" "$HERE/kingpod.py" wait "$POD" >/dev/null || { log "pod never served"; exit 3; }
REPL=$("$PY" -c 'import json; m=json.load(open("'"$HERE"'/state/pods.json"))["'"$POD"'"]; print(int((m.get("plan") or {}).get("replicas") or 1))')
export BENCH_API_KEY=$("$PY" -c 'import json; print(json.load(open("'"$HERE"'/state/pods.json"))["'"$POD"'"]["key"])')
TAG="${BUDGET_TAG:-}"   # BUDGET_TAG=4h250 -> the Albedo 4 h / 250-step budget as a separate cell
CELL="$INTO/king/swebench-verified${TAG:+@$TAG}__t0"
[ -z "$TAG" ] && [ -d "$CELL" ] && mv "$CELL" "$INTO/king/swebench-verified__t0.contended-$(date -u +%H%M)"
BUDGET_ARGS=(--agent-timeout-s 3600); [ "$TAG" = 4h250 ] && BUDGET_ARGS=(--budget-tag 4h250 --agent-timeout-s 14400 --step-limit 250)
log "$LABEL: $POD ($REPL replica(s)) -> SWE-bench Verified on Daytona at $((64*REPL)) in flight"
"$PY" "$HERE/harbor_cell.py" run --env swebench-verified --model "$("$PY" -c 'import json; print(json.load(open("'"$HERE"'/state/pods.json"))["'"$POD"'"]["served"])')" --model-label king \
  --model-url "$("$PY" -c 'import json; print(json.load(open("'"$HERE"'/state/pods.json"))["'"$POD"'"]["base_url"])')" --model-key-env BENCH_API_KEY --out "$INTO/king" --concurrency "${SWE_INFLIGHT:-$((64*REPL))}" "${BUDGET_ARGS[@]}"
"$PY" "$HERE/publish.py" --run-dir "$INTO" --only-cells "king/swebench-verified${TAG:+@$TAG}__t0"
log "$LABEL: done"
