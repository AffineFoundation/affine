#!/bin/bash
# Resume an interrupted / infra-cut Terminal-Bench 2 job on a fresh H200/B200: finished trials stay, trials that
# never reached the model (Daytona start-timeouts, harbor errors, connection errors) re-run, cell republished.
#   tb2_resume.sh <ref: sha256 | hf://repo@rev> <label> <run_id>
# 2026-09-24: the tb2 "resume" the watcher had was a fresh pass that skipped on the existing summary (3 pods wasted).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"; PY="${BENCHSUITE_PYTHON:-$REPO/.venv/bin/python}"; BENCH_HOME="${BENCH_HOME:-$HOME/benchsuite}"
source "$HERE/env.sh"
export LIUM_API_KEY="${LIUM_API_KEY:-${LIUM:-}}"; export HARBOR_BIN="${HARBOR_BIN:-$BENCH_HOME/harborenv/bin/harbor}"
[ -n "${DAYTONA_API_KEY:-}" ] || export DAYTONA_API_KEY=$(PATH=$HOME/.local/bin:$PATH op read --no-newline "op://Arbos/fywmj6vtq5delybw5c7a53l2qa/notesPlain" 2>/dev/null | grep -o 'dtn_[A-Za-z0-9_-]*' | head -1)
REF="$1" LABEL="$2" INTO="$BENCH_HOME/runs/$3"
log() { echo "[tb2-resume] $(date -u +%FT%TZ) $*"; }
CELL="$INTO/king/terminal-bench-2__t0"
[ -f "$CELL/harbor/config.json" ] || { log "$LABEL: no resumable harbor job under $CELL"; exit 2; }
if [[ "$REF" == hf://* ]]; then DIGEST="${REF##*@}"; EXTRA=(--hf "${REF#hf://}"); else DIGEST="$REF"; EXTRA=(); fi
POD=""; T0=$(date +%s)
while [ -z "$POD" ] && [ $(( $(date +%s) - T0 )) -lt 14400 ]; do
  for PLAN in ${TB2_PLANS:-h200-1x b200-1x}; do POD=$("$PY" "$HERE/kingpod.py" rent --plan "$PLAN" --digest "$DIGEST" "${EXTRA[@]}" --expected-hours 8 2>/dev/null | tail -1) && [[ "$POD" == bench-king-* ]] && break; POD=""; done
  [ -n "$POD" ] || { log "$LABEL: no H200/B200 stock; retry in 3 min"; sleep 180; }
done
[ -n "$POD" ] || { log "$LABEL: no stock within the rent deadline"; exit 2; }
trap '"$PY" "$HERE/kingpod.py" release "$POD" >/dev/null 2>&1' EXIT
"$PY" "$HERE/kingpod.py" wait "$POD" >/dev/null || { log "pod never served"; exit 3; }
podf() { "$PY" -c 'import json,sys; m=json.load(open("'"$HERE"'/state/pods.json"))[sys.argv[1]]; print(m[sys.argv[2]])' "$1" "$2"; }
export BENCH_API_KEY=$(podf "$POD" key)
"$PY" - "$CELL/harbor" "$(podf "$POD" base_url)" "$(podf "$POD" served)" <<'PY'
import json, sys
from pathlib import Path
job, url, served = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
def patch(o):
    ch = False
    if isinstance(o, dict):
        for k, v in list(o.items()):
            # Terminus keeps the model URL under `api_base`; mini-swe-agent under OPENAI_API_BASE
            if k in ("OPENAI_API_BASE", "OPENAI_BASE_URL", "ANTHROPIC_BASE_URL", "api_base", "base_url") and isinstance(v, str) and v != url: o[k] = url; ch = True
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
m = 0
INFRA = ("EnvironmentStartTimeoutError", "EnvironmentBuildError", "SandboxError", "SandboxBuildFailedError", "SandboxTimeoutError",
         "NetworkConnectionError", "ApiRateLimitError", "CancelledError", "InternalServerError", "RuntimeError")
for rp in job.glob("*/result.json"):
    r = json.loads(rp.read_text()); exc = r.get("exception_info") or {}; et = exc.get("exception_type") or ""
    if et in INFRA or "Sandbox" in et or any(w in (exc.get("exception_message") or "") for w in ("Connection error", "connection error", "Agent install failed")):
        rp.rename(rp.with_suffix(".json.infra")); m += 1
print(f"[tb2-resume] {n} config files re-pointed at {url}; {m} infra trials queued for re-run")
PY
log "$LABEL: $POD -> resuming $CELL (${TB2_INFLIGHT:-16} in flight, official per-task timeouts)"
"$PY" "$HERE/harbor_cell.py" resume --env terminal-bench-2 --model "$(podf "$POD" served)" --model-label king --model-url "$(podf "$POD" base_url)" --model-key-env BENCH_API_KEY --out "$INTO/king" --concurrency "${TB2_INFLIGHT:-16}" --agent-timeout-s 3600
"$PY" "$HERE/publish.py" --run-dir "$INTO" --only-cells "king/terminal-bench-2__t0"
log "$LABEL: done"
