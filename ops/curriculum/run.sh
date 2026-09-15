#!/bin/bash
# pm2 / one-off wrapper for the curriculum job on the validator box. Env
# handling as ops/improvement-loop/run.sh: the R2 keys and the Discord bot
# token come from the frozen snapshot ~/.affine-validator.env or the repo
# .env (KEY=VALUE with pm2 noise) -- parsed, never sourced, never printed.
#
#   pm2 start ops/curriculum/run.sh --name affine-curriculum --interpreter bash \
#       --cron-restart "20 15 * * *" --no-autorestart
#   ops/curriculum/run.sh --no-publish --no-discord      # dry run
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$REPO/.venv/bin/python"
for ENV_FILE in "${AFFINE_VALIDATOR_ENV:-$HOME/.affine-validator.env}" "$REPO/.env"; do
  [[ -r "$ENV_FILE" ]] || continue
  eval "$("$PY" - "$ENV_FILE" <<'PY'
import os, re, shlex, sys
from pathlib import Path

valid = re.compile(r"^[A-Z][A-Z0-9_]*$")
want = ("DISCORD_BOT_TOKEN_ARBOS_BITTENSOR", "DATA_R2_ACCESS_KEY_ID",
        "DATA_R2_SECRET_ACCESS_KEY", "DATA_R2_ENDPOINT")
for line in Path(sys.argv[1]).read_text().splitlines():
    line = line.strip("\n")
    if not line or line.lstrip().startswith("#") or "=" not in line:
        continue
    key, val = line.split("=", 1)
    key = key.removeprefix("export ").strip()
    if key in want and valid.match(key) and not os.environ.get(key):
        print(f"export {key}={shlex.quote(val)}")
PY
)"
done
cd "$REPO"
exec "$PY" ops/curriculum/run.py "$@"
