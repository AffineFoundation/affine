#!/bin/bash
# Shared env loader for the benchsuite entry points (run.sh = the pm2 watcher,
# pass.sh = one pass by hand). KEY=VALUE lines from ~/.affine-validator.env, the
# repo .env and ~/.affine-op.env, exported through a shell-safe parser (never
# `source` them directly; values may contain spaces). Expects HERE, REPO, PY.
for ENV_FILE in "${AFFINE_VALIDATOR_ENV:-$HOME/.affine-validator.env}" "$REPO/.env" "$HOME/.affine-op.env"; do
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

export PATH="$HOME/.local/bin:$PATH" BENCHSUITE_PYTHON="$PY" BENCHSUITE_FP_WORKDIR="$HOME/benchsuite/tmp" BENCHSUITE_CHAT_IMAGE="affine-bench-chat:py311"
mkdir -p "$HOME/benchsuite/tmp"
# the box env snapshot names the Prime key PRIME; the Prime CLI/SDK want PRIME_API_KEY
export PRIME_API_KEY="${PRIME_API_KEY:-${PRIME:-}}"
export LIUM_API_KEY="${LIUM_API_KEY:-${LIUM:-}}"
export AFFINE_EVAL_R2_ENDPOINT="${AFFINE_EVAL_R2_ENDPOINT:-${R2_ENDPOINT:-}}"
