#!/bin/bash
# Source this from a pm2 wrapper. Exports operator secrets from the frozen
# snapshot ~/.affine-validator.env (KEY=VALUE, often unquoted, with pm2
# metadata mixed in) and, for keys still missing, from the repo .env — never
# `source` those files directly. Same approach as ops/evalwatch/run_evalwatch.sh
# and ops/kingboard/run.sh.
#
#   REPO=... ; source "$REPO/ops/health/pm2_env.sh"          # the guards' few keys
#   REPO=... ; PM2_ENV_ALL=1 source "$REPO/ops/health/pm2_env.sh"  # every UPPERCASE secret (the fold)
: "${REPO:?REPO must be set before sourcing pm2_env.sh}"
for _ENV_FILE in "${AFFINE_VALIDATOR_ENV:-$HOME/.affine-validator.env}" "$REPO/.env"; do
  [[ -r "$_ENV_FILE" ]] || continue
  eval "$("$REPO/.venv/bin/python" - "$_ENV_FILE" "${PM2_ENV_ALL:-0}" <<'PY'
import os, re, shlex, sys
from pathlib import Path

valid = re.compile(r"^[A-Z][A-Z0-9_]*$")
want = {"AFFINE_EVAL_TOKEN", "LIUM_API_KEY", "DISCORD_BOT_TOKEN_ARBOS_BITTENSOR",
        "DATA_R2_ACCESS_KEY_ID", "DATA_R2_SECRET_ACCESS_KEY", "DATA_R2_ENDPOINT", "R2_ENDPOINT"}
# shell-owned names a process-env dump also carries: never re-export them
skip = {"PATH", "HOME", "PWD", "OLDPWD", "SHELL", "SHLVL", "TERM", "USER", "LOGNAME",
        "LANG", "LC_ALL", "VIRTUAL_ENV", "VIRTUAL_ENV_PROMPT", "PYTHONPATH", "PS1",
        "XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS", "SSH_AUTH_SOCK", "SSH_CLIENT",
        "SSH_CONNECTION", "PM2_HOME", "PM2_USAGE", "PM2_JSON_PROCESSING", "NODE_APP_INSTANCE",
        "LS_COLORS", "LESSOPEN", "LESSCLOSE", "BROWSER", "FORCE_COLOR", "NO_COLOR"}
everything = sys.argv[2] == "1"
for line in Path(sys.argv[1]).read_text().splitlines():
    line = line.strip("\n")
    if not line or line.lstrip().startswith("#") or "=" not in line:
        continue
    key, val = line.split("=", 1)
    key = key.strip().removeprefix("export ").strip()
    if not valid.match(key) or key in skip or os.environ.get(key):
        continue
    if everything or key in want:
        print(f"export {key}={shlex.quote(val.strip())}")
PY
)"
done
unset _ENV_FILE
