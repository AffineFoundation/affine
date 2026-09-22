#!/bin/bash
# pm2 / one-off wrapper for the improvement-loop attribution job on the
# validator box. Same env handling as ops/evalwatch/run_evalwatch.sh: the
# Discord bot token comes from the frozen snapshot ~/.affine-validator.env or
# the repo .env (KEY=VALUE with pm2 noise) -- parsed, never sourced.
#
#   pm2 start ops/improvement-loop/run.sh --name affine-improvement-loop \
#       --interpreter bash -- watch --interval 300
#   ops/improvement-loop/run.sh once --run <run_id> [--against <run_id>] [--discord]
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$REPO/.venv/bin/python"
for ENV_FILE in "${AFFINE_VALIDATOR_ENV:-$HOME/.affine-validator.env}" "$REPO/.env"; do
  [[ -r "$ENV_FILE" ]] || continue
  eval "$("$PY" - "$ENV_FILE" <<'PY'
import re, shlex, sys
from pathlib import Path

valid = re.compile(r"^[A-Z][A-Z0-9_]*$")
want = ("DISCORD_BOT_TOKEN_ARBOS_BITTENSOR", "BENCHSUITE_RUNS")
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
cd "$REPO/ops/improvement-loop"
mode="${1:-watch}"; shift || true
case "$mode" in
  watch) exec "$PY" watch.py "$@" ;;
  once)  exec "$PY" attribution.py "$@" ;;
  *)     echo "usage: $0 {watch|once} [args]" >&2; exit 2 ;;
esac
