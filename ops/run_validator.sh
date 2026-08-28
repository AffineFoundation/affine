#!/bin/bash
# pm2 wrapper for the affine validator. Doppler auth has been broken since
# the 2026-08-19 machine reset, so secrets come from a frozen env snapshot
# (~/.affine-validator.env, chmod 600, outside the repo) taken from the
# last known-good process env with LIUM_API_KEY refreshed from
# ~/.lium/config.ini [api]. Once doppler login is restored this can go
# back to the `doppler run` wrapper in scripts/ecosystem.config.js.
#
# The snapshot is KEY=VALUE (often unquoted, and may include pm2 metadata).
# Never `source` it directly — values like SSH_CONNECTION contain spaces and
# bash word-splits them into bogus commands.
set -euo pipefail
ENV_FILE="${AFFINE_VALIDATOR_ENV:-/home/const/.affine-validator.env}"
if [[ ! -r "$ENV_FILE" ]]; then
  echo "run_validator.sh: missing readable env snapshot: $ENV_FILE" >&2
  exit 1
fi
eval "$(/home/const/subnet120/.venv/bin/python - "$ENV_FILE" <<'PY'
import re, shlex, sys
from pathlib import Path

path = Path(sys.argv[1])
valid = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# pm2 / shell noise from a process-env dump — do not export
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
    line = line.strip("\n")
    if not line or line.lstrip().startswith("#") or "=" not in line:
        continue
    key, val = line.split("=", 1)
    if key in skip or not valid.match(key):
        continue
    print(f"export {key}={shlex.quote(val)}")
PY
)"
cd /home/const/subnet120/affine
exec /home/const/subnet120/.venv/bin/python -m affine.validator
