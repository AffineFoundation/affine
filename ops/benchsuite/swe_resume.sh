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
log() { echo "[swe-resume] $(date -u +%FT%TZ) $*"; }
CELL="$INTO/king/swebench-verified${TAG:+@$TAG}__t0"
[ -d "$CELL/harbor" ] || { log "$LABEL: no harbor job under $CELL"; exit 2; }
POD=""; for PLAN in ${SWE_PLANS:-h200-2x b200-2x h200-1x b200-1x pro6000-1x}; do POD=$("$PY" "$HERE/kingpod.py" rent --plan "$PLAN" --digest "$DIGEST" 2>/dev/null | tail -1) && [ -n "$POD" ] && break; POD=""; done
[ -n "$POD" ] || { log "$LABEL: no stock"; exit 2; }
trap '"$PY" "$HERE/kingpod.py" release "$POD" >/dev/null 2>&1' EXIT
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
log "$LABEL: $POD ($REPL replica(s)) -> resuming $CELL at $((64*REPL)) in flight"
"$PY" "$HERE/harbor_cell.py" resume --env swebench-verified ${TAG:+--budget-tag $TAG} --model "$(podf "$POD" served)" --model-label king \
  --model-url "$(podf "$POD" base_url)" --model-key-env BENCH_API_KEY --out "$INTO/king" --concurrency $((64*REPL)) --agent-timeout-s 3600
"$PY" "$HERE/publish.py" --run-dir "$INTO" --only-cells "king/swebench-verified${TAG:+@$TAG}__t0"
log "$LABEL: done"
