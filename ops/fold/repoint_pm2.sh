#!/usr/bin/env bash
# Re-point pm2 `affine-corpus-refresh` at ops/fold/run_fold.sh (the wrapper
# that records exit / traceback and pages), keeping whatever cron expression
# the process currently has. Waits for a running fold to exit first; the
# `pm2 start` that follows does not fold immediately (SKIP_NEXT flag), the
# next cron slot does.
#
#   bash ops/fold/repoint_pm2.sh            # idempotent; safe to re-run
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NAME=affine-corpus-refresh
PY="$REPO/.venv/bin/python"
cd "$REPO"
ts() { date -u +%FT%TZ; }

while pgrep -f "python ops/corpus_build.py" >/dev/null; do
  echo "$(ts) a fold is running; waiting"; sleep 30
done

INFO=$(pm2 jlist 2>/dev/null | "$PY" -c '
import json, sys
name = sys.argv[1]
for p in json.load(sys.stdin):
    if p.get("name") == name:
        e = p["pm2_env"]
        print(json.dumps({"cron": e.get("cron_restart") or "", "script": e.get("pm_exec_path") or ""})); break
else:
    print("{}")' "$NAME")
CRON=$("$PY" -c 'import json,sys; print(json.loads(sys.argv[1]).get("cron",""))' "$INFO")
SCRIPT=$("$PY" -c 'import json,sys; print(json.loads(sys.argv[1]).get("script",""))' "$INFO")
# a cron expression has five fields; anything else (missing, or a mangled
# value like "0") falls back to the fold worker's 6-h schedule
if [[ $(wc -w <<<"$CRON") -ne 5 ]]; then
  echo "$(ts) cron on $NAME is '$CRON' (not 5 fields) -> using ${FOLD_CRON:-0 */6 * * *}"
  CRON="${FOLD_CRON:-0 */6 * * *}"; FORCE=1
fi
if [[ "$SCRIPT" == *"run_fold.sh" && "${FORCE:-0}" != 1 ]]; then
  echo "$(ts) $NAME already runs ops/fold/run_fold.sh (cron $CRON); nothing to do"; exit 0
fi
echo "$(ts) re-pointing $NAME: script $SCRIPT -> ops/fold/run_fold.sh, cron kept: $CRON"
touch "$REPO/ops/fold/SKIP_NEXT"
pm2 delete "$NAME" >/dev/null 2>&1 || true
pm2 start ops/fold/run_fold.sh --name "$NAME" --interpreter bash --cron "$CRON" --no-autorestart \
    -o "$REPO/affine/logs/corpus_refresh.out.log" -e "$REPO/affine/logs/corpus_refresh.err.log" >/dev/null
for _ in $(seq 1 30); do [[ -e "$REPO/ops/fold/SKIP_NEXT" ]] || break; sleep 2; done   # the immediate run consumed the flag
rm -f "$REPO/ops/fold/SKIP_NEXT"
pm2 save >/dev/null
pm2 describe "$NAME" | grep -E "script path|script args|cron restart|status|out log path"
echo "$(ts) done: $NAME now execs ops/fold/run_fold.sh on cron $CRON"
