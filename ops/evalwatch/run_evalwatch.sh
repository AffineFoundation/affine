#!/bin/bash
# pm2 wrapper for affine-evalwatch. Same env handling as ops/run_validator.sh:
# secrets come from the frozen snapshot ~/.affine-validator.env (KEY=VALUE,
# often unquoted, with pm2 metadata mixed in) — never `source` it directly.
#
#   pm2 start ops/evalwatch/run_evalwatch.sh --name affine-evalwatch \
#       --interpreter bash -- --interval 300
#   pm2 start ops/evalwatch/run_evalwatch.sh --name affine-evalwatch \
#       --interpreter bash -- --dry-run          # log alerts, never post
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${AFFINE_VALIDATOR_ENV:-$HOME/.affine-validator.env}"
if [[ -r "$ENV_FILE" ]]; then
  eval "$("$REPO/.venv/bin/python" - "$ENV_FILE" <<'PY'
import re, shlex, sys
from pathlib import Path

valid = re.compile(r"^[A-Z][A-Z0-9_]*$")
want = ("AFFINE_EVAL_TOKEN", "LIUM_API_KEY", "DISCORD_BOT_TOKEN_ARBOS_BITTENSOR")
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
fi
cd "$REPO/ops/evalwatch"
exec "$REPO/.venv/bin/python" evalwatch.py "$@"
