#!/bin/bash
# pm2 wrapper for the Discord mirror (pm2 name: affine-discord-mirror).
#
# Loads secrets the same way ops/run_validator.sh does: doppler is broken on
# the box, so the env comes from the frozen snapshots ~/.affine-validator.env
# and the repo .env. Neither is `source`d directly (values with spaces would
# word-split); a small parser exports KEY=VALUE pairs only.
#
#   pm2 start ops/discord-mirror/run_mirror.sh --name affine-discord-mirror
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="$REPO/.venv/bin/python"
[[ -x "$PY" ]] || PY="$(command -v python3)"

ENV_FILES=("${AFFINE_VALIDATOR_ENV:-$HOME/.affine-validator.env}" "$REPO/.env")
for f in "${ENV_FILES[@]}"; do
  [[ -r "$f" ]] || continue
  eval "$("$PY" - "$f" <<'PY'
import os, re, shlex, sys
from pathlib import Path

valid = re.compile(r"^[A-Z][A-Z0-9_]*$")
# the validator snapshot is a process-env dump: never clobber the live shell
# (PATH, HOME, PWD, ...) or import pm2 metadata; only add what is missing.
for line in Path(sys.argv[1]).read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    line = line.removeprefix("export ").strip()
    key, val = line.split("=", 1)
    key = key.strip()
    if not valid.match(key) or key in os.environ or key.startswith(("PM2_", "NODE_")):
        continue
    val = val.strip()
    if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
        val = val[1:-1]
    print(f"export {key}={shlex.quote(val)}")
PY
)"
done

cd "$HERE"
exec "$PY" mirror.py run "$@"
