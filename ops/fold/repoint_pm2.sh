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

read -r CRON SCRIPT <<<"$(pm2 jlist 2>/dev/null | "$PY" -c '
import json, sys
name = sys.argv[1]
for p in json.load(sys.stdin):
    if p.get("name") == name:
        e = p["pm2_env"]
        print(e.get("cron_restart") or "-", e.get("pm_exec_path") or "-"); break
else:
    print("- -")' "$NAME")"
[[ "$CRON" == "-" ]] && CRON="0 */6 * * *"   # fold worker's 6-h schedule (docs/auto-research-loop.md §2.1)
if [[ "$SCRIPT" == *"run_fold.sh" ]]; then
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
