#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# uv workspace: affine installed editable, research/ops deps-only.
# One root .venv + uv.lock. GPU [eval] extras are never pulled here (no GPU on this box).
uv sync --all-packages

# Single shared env file: `harness` / `scripts` import from research/
echo "export PYTHONPATH=\"$PWD/research\${PYTHONPATH:+:\$PYTHONPATH}\"" > .env

cat <<MSG

OK. Activate with:
  cd ~/subnet120 && source .venv/bin/activate && source .env

MSG
