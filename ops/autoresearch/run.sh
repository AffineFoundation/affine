#!/bin/bash
# pm2 wrapper for the auto-research daemon (docs/auto-research-loop.md).
#   pm2 start ops/autoresearch/run.sh --name affine-autoresearch --interpreter bash -- --interval 600
# Env parsed (never sourced) from ~/.affine-validator.env and the repo .env: Engy key, Discord token.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$REPO/.venv/bin/python"
for ENV_FILE in "${AFFINE_VALIDATOR_ENV:-$HOME/.affine-validator.env}" "$REPO/.env"; do
  [[ -r "$ENV_FILE" ]] || continue
  eval "$("$PY" - "$ENV_FILE" <<'PY'
import re, shlex, sys
from pathlib import Path
valid = re.compile(r"^[A-Z][A-Z0-9_]*$")
want = ("DISCORD_BOT_TOKEN_ARBOS_BITTENSOR", "ENGY_EVAL", "ENGY_2", "OPENROUTER_API_KEY", "HF_TOKEN")
for line in Path(sys.argv[1]).read_text().splitlines():
    line = line.strip("\n")
    if not line or line.lstrip().startswith("#") or "=" not in line:
        continue
    key, val = line.split("=", 1)
    key = key.removeprefix("export ").strip()
    if key in want and valid.match(key):
        print(f"export {key}={shlex.quote(val)}")
PY
)"
done
cd "$REPO/ops/autoresearch"
exec "$PY" daemon.py "$@"
