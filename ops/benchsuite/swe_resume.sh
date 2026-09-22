#!/bin/bash
# Resume an interrupted SWE-bench Verified Daytona job (harbor_cell.py resume) on a fresh
# serving pod: finished trials stay, the unfinished ones re-run, then the cell is published.
#   swe_resume.sh <digest> <label> <run_id> [budget_tag]
# 2026-09-20: reign 16's job died at 468/504 trials when its pass was killed.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"; PY="${BENCHSUITE_PYTHON:-$REPO/.venv/bin/python}"; BENCH_HOME="${BENCH_HOME:-$HOME/benchsuite}"
source "$HERE/env.sh"
export LIUM_API_KEY="${LIUM_API_KEY:-${LIUM:-}}"; export HARBOR_BIN="${HARBOR_BIN:-$BENCH_HOME/harborenv/bin/harbor}"
[ -n "${DAYTONA_API_KEY:-}" ] || export DAYTONA_API_KEY=$(op read --no-newline "op://Arbos/fywmj6vtq5delybw5c7a53l2qa/notesPlain" 2>/dev/null | grep -o 'dtn_[A-Za-z0-9_-]*' | head -1)
DIGEST="$1" LABEL="$2" INTO="$BENCH_HOME/runs/$3" TAG="${4:-}"
AGENT_TIMEOUT_S=3600; [ "$TAG" = 4h250 ] && AGENT_TIMEOUT_S=14400
log() { echo "[swe-resume] $(date -u +%FT%TZ) $*"; }
CELL="$INTO/king/swebench-verified${TAG:+@$TAG}__t0"
[ -d "$CELL/harbor" ] || { log "$LABEL: no harbor job under $CELL"; exit 2; }
POD=""; T0=$(date +%s)
while [ $(( $(date +%s) - T0 )) -lt "${SWE_RENT_DEADLINE_S:-14400}" ]; do
  for PLAN in ${SWE_PLANS:-h200-2x b200-2x h200-1x b200-1x pro6000-1x}; do POD=$("$PY" "$HERE/kingpod.py" rent --plan "$PLAN" --digest "$DIGEST" 2>/dev/null | tail -1) && [ -n "$POD" ] && break; POD=""; done
  [ -n "$POD" ] && break; log "$LABEL: no stock; retry in 3 min"; sleep 180
done
[ -n "$POD" ] || { log "$LABEL: no stock within the rent deadline"; exit 2; }
trap '"$PY" "$HERE/kingpod.py" release "$POD" >/dev/null 2>&1' EXIT
# lifetime for the pod reaper from the job's own budget: trials x agent budget / in flight, x1.3 slack, + 2 h
# (2026-09-21: 27-h @4h250 jobs lost their serving box at kingpod's 14-h default -> NetworkConnectionError x 270)
JOB_H=$(python3 -c "import math; print(max(14, math.ceil(500 * ${AGENT_TIMEOUT_S:-3600} / ${SWE_INFLIGHT_EST:-64} / 3600 * 1.3) + 2))")
"$PY" -c "import sys; sys.path.insert(0, '$REPO/ops/pods'); import registry; registry.register('$POD', expected_hours=$JOB_H, source='explicit', meta={'job': 'swebench ${TAG:-1h} rerun/resume'})" 2>/dev/null || true
log "pod registered for $JOB_H h (budget ${AGENT_TIMEOUT_S:-3600}s x 500 / ${SWE_INFLIGHT_EST:-64} in flight)"
"$PY" "$HERE/kingpod.py" wait "$POD" >/dev/null || { log "pod never served"; exit 3; }
podf() { "$PY" -c 'import json,sys; m=json.load(open("'"$HERE"'/state/pods.json"))[sys.argv[1]]; print(m[sys.argv[2]])' "$1" "$2"; }
export BENCH_API_KEY=$(podf "$POD" key)
REPL=$("$PY" -c 'import json; m=json.load(open("'"$HERE"'/state/pods.json"))["'"$POD"'"]; print(int((m.get("plan") or {}).get("replicas") or 1))')
# the resumed job must point at the NEW pod: rewrite the endpoint in the saved harbor configs
"$PY" - "$CELL/harbor" "$(podf "$POD" base_url)" "$(podf "$POD" served)" <<'PY'
import json, sys
from pathlib import Path
job, url, served = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
def patch(o):
    ch = False
    if isinstance(o, dict):
        for k, v in list(o.items()):
            if k in ("OPENAI_API_BASE", "OPENAI_BASE_URL") and isinstance(v, str) and v != url: o[k] = url; ch = True
            elif k == "model_name" and isinstance(v, str) and v.startswith("openai/") and v != f"openai/{served}": o[k] = f"openai/{served}"; ch = True
            else: ch |= patch(v)
    elif isinstance(o, list):
        for v in o: ch |= patch(v)
    return ch
n = 0
for fp in [job / "config.json", job / "lock.json", *job.glob("*/config.json"), *job.glob("*/lock.json")]:
    try: c = json.loads(fp.read_text())
    except (OSError, ValueError): continue
    if patch(c): fp.write_text(json.dumps(c, indent=2)); n += 1
print(f"[swe-resume] {n} harbor config/lock files re-pointed at {url}")
PY
# infra-errored trials (Daytona NetworkConnectionError / provisioning) are "finished" for harbor and would not
# re-run: drop their result.json so `job resume` picks them up again (RETRY_INFRA=0 keeps them)
if [ "${RETRY_INFRA:-1}" = 1 ]; then
  "$PY" - "$CELL/harbor" <<'PY'
import json, sys
from pathlib import Path
n = 0
for rp in Path(sys.argv[1]).glob("*/result.json"):
    try: r = json.loads(rp.read_text())
    except ValueError: continue
    exc = r.get("exception_info") or {}
    et = exc.get("exception_type") or ""
    if not et:
        continue
    # same rule as harbor_cell.is_infra_env: our box / Daytona / harbor failed, or the trial never got a model token
    if et in ("NetworkConnectionError", "SandboxError", "SandboxBuildFailedError", "SandboxTimeoutError", "EnvironmentStartTimeoutError",
              "EnvironmentBuildError", "ApiRateLimitError", "CancelledError", "DaytonaError") or "Provision" in et or "Sandbox" in et \
       or any(m in (exc.get("exception_message") or "") for m in ("Connection refused", "Max retries exceeded", "APIConnectionError",
              "502 Bad Gateway", "503 Service", "504 Gateway", "Agent install failed", "Failed to execute session command")) \
       or not ((r.get("agent_result") or {}).get("n_input_tokens") or 0):
        rp.rename(rp.with_suffix(".json.infra")); n += 1
print(f"[swe-resume] {n} infra-errored trials queued for re-run")
PY
fi
INFLIGHT="${SWE_INFLIGHT:-$((64*REPL))}"
BUDGET_ARGS=(--agent-timeout-s "$AGENT_TIMEOUT_S"); [ "$TAG" = 4h250 ] && BUDGET_ARGS+=(--step-limit 250)
log "$LABEL: $POD ($REPL replica(s)) -> resuming $CELL at $INFLIGHT in flight (budget ${TAG:-1h})"
"$PY" "$HERE/harbor_cell.py" resume --env swebench-verified ${TAG:+--budget-tag $TAG} --model "$(podf "$POD" served)" --model-label king \
  --model-url "$(podf "$POD" base_url)" --model-key-env BENCH_API_KEY --out "$INTO/king" --concurrency "$INFLIGHT" "${BUDGET_ARGS[@]}"
"$PY" "$HERE/publish.py" --run-dir "$INTO" --only-cells "king/swebench-verified${TAG:+@$TAG}__t0"
log "$LABEL: done"
