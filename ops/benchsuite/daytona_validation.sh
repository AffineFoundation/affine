#!/bin/bash
# Validation pass for the Daytona route (Jacob 2026-09-17): SWE-bench Verified at
# Albedo's budget — 4 h wall / 250 model calls per task (suite.toml
# [sandbox_daytona.budgets.4h250]) — for reign 13 and the teacher, in parallel,
# published as the budget-tagged cell swebench-verified@4h250 next to (never over)
# the 1-h cells.
#   king    reign 13 served on a rented Lium pod (b200-1x first; the pod only serves)
#   teacher Qwen/Qwen3.8-27B via Engy (hosted, reachable from the Daytona sandboxes;
#           serving stack not ours -> recorded as served_by on the cell)
#   bash daytona_validation.sh [king|teacher|both]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="${BENCHSUITE_PYTHON:-$REPO/.venv/bin/python}"
BENCH_HOME="${BENCH_HOME:-$HOME/benchsuite}"
# shellcheck disable=SC1091
source "$HERE/env.sh"
export LIUM_API_KEY="${LIUM_API_KEY:-${LIUM:-}}"
export HARBOR_BIN="${HARBOR_BIN:-$BENCH_HOME/harborenv/bin/harbor}"
WHAT="${1:-both}"
log() { echo "[daytona-validation] $(date -u +%FT%TZ) $*"; }
toml() { "$PY" -c 'import tomllib,sys; d=tomllib.load(open("'"$HERE"'/suite.toml","rb")); v=d
for k in sys.argv[1].split("."): v=v[k]
print(",".join(v) if isinstance(v,list) else v)' "$1"; }
if [ -z "${DAYTONA_API_KEY:-}" ]; then
  export DAYTONA_API_KEY; DAYTONA_API_KEY=$(op read --no-newline "${DAYTONA_OP_ITEM:-op://Arbos/fywmj6vtq5delybw5c7a53l2qa/notesPlain}" 2>/dev/null | grep -o 'dtn_[A-Za-z0-9_-]*' | head -1)
fi
[ -n "${DAYTONA_API_KEY:-}" ] || { log "no DAYTONA_API_KEY"; exit 1; }
TAG=4h250
TIMEOUT_S=$(toml sandbox_daytona.budgets.4h250.agent_timeout_s)
STEPS=$(toml sandbox_daytona.budgets.4h250.step_limit)
CONC=$(toml sandbox_daytona.concurrency)
R13=6d0ee567e33ee44fc238cc4a1eaae3ad9c98e3e63669548bc2fa75b94c5ebb47
R13_RUN=20260915T0753Z-6d0ee567e33e
TEACHER_RUN=20260912T1455Z-0ce59769300c

teacher() {
  [ -n "${ENGY:-}" ] || { log "no ENGY key in the env"; return 1; }
  log "teacher: SWE-bench Verified @$TAG on Daytona via Engy (qwen3.8-27b), concurrency $CONC"
  "$PY" "$HERE/harbor_cell.py" run --env swebench-verified --budget-tag "$TAG" --agent-timeout-s "$TIMEOUT_S" --step-limit "$STEPS" \
      --concurrency "$CONC" --model qwen3.8-27b --model-label teacher --model-url https://api.engy.ai/v1 --model-key-env ENGY \
      --out "$BENCH_HOME/runs/$TEACHER_RUN/teacher"
  "$PY" - "$BENCH_HOME/runs/$TEACHER_RUN/teacher/swebench-verified@$TAG__t0/summary.json" <<'PY'
import json, sys
p = sys.argv[1]; s = json.load(open(p))
s["served_by"] = {"provider": "Engy (hosted)", "base_url": "https://api.engy.ai/v1", "model": "qwen3.8-27b",
                  "note": "hosted serving stack (parsers / context / batching not ours); the 1-h teacher cell was served by our vLLM 0.28.0 stack"}
json.dump(s, open(p, "w"), indent=1)
PY
  "$PY" "$HERE/publish.py" --run-dir "$BENCH_HOME/runs/$TEACHER_RUN" --only-cells "teacher/swebench-verified@$TAG__t0" || log "teacher publish failed"
}

king() {
  local POD=""
  cleanup() { [ -n "$POD" ] && { log "releasing $POD"; "$PY" "$HERE/kingpod.py" release "$POD" || true; }; }
  trap cleanup EXIT
  for PLAN in b200-1x h200-1x pro6000-1x; do
    POD=$("$PY" "$HERE/kingpod.py" rent --plan "$PLAN" --digest "$R13" | tail -1) && [ -n "$POD" ] && break
    log "no pod on $PLAN"; POD=""
  done
  [ -n "$POD" ] || { log "no Lium stock for the king"; return 2; }
  "$PY" "$HERE/kingpod.py" wait "$POD" > /dev/null || { log "pod never served"; return 3; }
  local URL KEY SERVED
  URL=$("$PY" -c 'import json; print(json.load(open("'"$HERE"'/state/pods.json"))["'"$POD"'"]["base_url"])')
  KEY=$("$PY" -c 'import json; print(json.load(open("'"$HERE"'/state/pods.json"))["'"$POD"'"]["key"])')
  SERVED=$("$PY" -c 'import json; print(json.load(open("'"$HERE"'/state/pods.json"))["'"$POD"'"]["served"])')
  export BENCH_API_KEY="$KEY"
  log "king: reign 13 on $POD ($URL), SWE-bench Verified @$TAG on Daytona, concurrency $CONC"
  "$PY" "$HERE/harbor_cell.py" run --env swebench-verified --budget-tag "$TAG" --agent-timeout-s "$TIMEOUT_S" --step-limit "$STEPS" \
      --concurrency "$CONC" --model "$SERVED" --model-label king --model-url "$URL" --model-key-env BENCH_API_KEY \
      --out "$BENCH_HOME/runs/$R13_RUN/king"
  "$PY" - "$BENCH_HOME/runs/$R13_RUN/king/swebench-verified@$TAG__t0/summary.json" "$POD" "$HERE/state/pods.json" <<'PY'
import json, sys
p, pod, pods_path = sys.argv[1:]; s = json.load(open(p))
m = json.load(open(pods_path)).get(pod, {})
s["where"] = {"provider": "Lium (our fleet, TAO)", "pod_id": pod, "gpu": m.get("machine"), "plan": (m.get("plan") or {}).get("name"),
              "usd_per_hour": m.get("price"), "note": "pod serves the model only; task containers on Daytona"}
json.dump(s, open(p, "w"), indent=1)
PY
  "$PY" "$HERE/publish.py" --run-dir "$BENCH_HOME/runs/$R13_RUN" --only-cells "king/swebench-verified@$TAG__t0" || log "king publish failed"
}

case "$WHAT" in
  teacher) teacher ;;
  king) king ;;
  both) teacher & TP=$!; king; wait $TP ;;
esac
log "done"
