#!/bin/bash
# pm2 wrapper for the kingboard (see server.py). Loads the frozen env
# snapshot the same way ops/run_validator.sh does (doppler is broken on the
# box): KEY=VALUE lines from ~/.affine-validator.env and the repo .env,
# exported through a shell-safe parser (never `source` them directly).
# Only DATA_R2_* (read access to the affine-data bucket) is used here; with
# no key the builder falls back to the public https://data.affine.io mirror.
#
#   pm2 start ops/kingboard/run.sh --name affine-kingboard --interpreter bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="${KINGBOARD_PYTHON:-$REPO/.venv/bin/python}"
PORT="${KINGBOARD_PORT:-8790}"
HOST="${KINGBOARD_HOST:-127.0.0.1}"

for ENV_FILE in "${AFFINE_VALIDATOR_ENV:-$HOME/.affine-validator.env}" "$REPO/.env"; do
  [[ -r "$ENV_FILE" ]] || continue
  eval "$("$PY" - "$ENV_FILE" <<'PY'
import re, shlex, sys
from pathlib import Path

valid = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# pm2 / shell noise from a process-env dump — do not export (same list as
# ops/run_validator.sh, plus the login-shell basics)
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
    "PS1", "OLDPWD", "_", "SHLVL", "PWD", "HOME", "PATH", "SHELL", "USER",
    "TERM", "LOGNAME", "MAIL", "LANG",
}
for line in Path(sys.argv[1]).read_text().splitlines():
    line = line.strip("\n")
    if not line or line.lstrip().startswith("#") or "=" not in line:
        continue
    key, val = line.split("=", 1)
    key = key.strip().removeprefix("export ").strip()
    if key in skip or not valid.match(key):
        continue
    print(f"export {key}={shlex.quote(val.strip())}")
PY
)"
done

cd "$HERE"
exec "$PY" -m uvicorn server:app --host "$HOST" --port "$PORT" --log-level warning
