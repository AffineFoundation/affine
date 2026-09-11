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
skip = {"PS1", "OLDPWD", "_", "SHLVL", "PWD", "HOME", "PATH", "SHELL", "USER", "TERM"}
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
