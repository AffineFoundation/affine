#!/bin/bash
# pm2 / one-off wrapper for the per-reign king review on the validator box.
#
#   run_king_review.sh watch                 # the pm2 service (affine-king-review)
#   run_king_review.sh once [run_review.py args...]   # e.g. once --all --max-usd 90
#
# Env comes from the same frozen snapshot ops/run_validator.sh uses
# (~/.affine-validator.env, chmod 600, outside the repo): OPENROUTER_API_KEY
# is the judge key. The snapshot is KEY=VALUE with pm2 noise, so it is parsed
# the same way, never sourced. Everything the review writes lives under
# affine/state/king_review/ (trace cache, judgments, reports) and the fold's
# side-table dir affine/state/king_pivots/ -- both gitignored state.
set -euo pipefail
REPO="${AFFINE_REPO:-/home/const/subnet120}"
ENV_FILE="${AFFINE_VALIDATOR_ENV:-/home/const/.affine-validator.env}"
PY="$REPO/.venv/bin/python"
if [[ ! -r "$ENV_FILE" ]]; then
  echo "run_king_review.sh: missing readable env snapshot: $ENV_FILE" >&2
  exit 1
fi
eval "$("$PY" - "$ENV_FILE" <<'PY'
import re, shlex, sys
from pathlib import Path

path = Path(sys.argv[1])
valid = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
skip = {
    "name", "cwd", "exec_interpreter", "restart_delay", "kill_timeout",
    "merge_logs", "vizion", "autostart", "autorestart", "watch",
    "max_restarts", "instance_var", "pmx", "automation", "treekill",
    "username", "windowsHide", "kill_retry_time", "namespace",
    "pm_exec_path", "pm_cwd", "exec_mode", "pm_out_log_path",
    "pm_err_log_path", "pm_pid_path", "km_link", "vizion_running",
    "NODE_APP_INSTANCE", "PM2_USAGE", "PM2_JSON_PROCESSING", "PM2_HOME",
    "unique_id", "status", "pm_uptime", "created_at", "restart_time",
    "unstable_restarts", "version", "exit_code", "instances", "pm_id",
    "prev_restart_delay", "NODE_CHANNEL_FD", "NODE_CHANNEL_SERIALIZATION_MODE",
    "PS1", "OLDPWD", "_", "SHLVL",
}
for line in path.read_text().splitlines():
    if not line or line.lstrip().startswith("#") or "=" not in line:
        continue
    key, val = line.split("=", 1)
    if key in skip or not valid.match(key):
        continue
    print(f"export {key}={shlex.quote(val)}")
PY
)"
export KING_REVIEW_OUT="${KING_REVIEW_OUT:-$REPO/affine/state/king_review}"
export KING_REVIEW_CACHE="${KING_REVIEW_CACHE:-$REPO/affine/state/king_review/traces}"
mkdir -p "$KING_REVIEW_OUT" "$KING_REVIEW_CACHE"
cd "$REPO/ops/king-review"
COMMON=(--state-json "$REPO/affine/state/state.json"
        --merge-into-dir "$REPO/affine/state/king_pivots"
        --check-published --procs 16 --concurrency 8)
mode="${1:-watch}"; shift || true
case "$mode" in
  watch) exec "$PY" run_review.py --watch --daily-at 14:30 --poll-seconds 600 \
           --max-usd "${KING_REVIEW_DAILY_MAX_USD:-40}" "${COMMON[@]}" "$@" ;;
  once)  exec "$PY" run_review.py "${COMMON[@]}" "$@" ;;
  *)     echo "usage: $0 {watch|once} [run_review.py args]" >&2; exit 2 ;;
esac
