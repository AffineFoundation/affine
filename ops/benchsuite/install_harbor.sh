#!/bin/bash
# Harbor CLI + Daytona extra for the cloud-sandbox route (harbor_cell.py), pinned by
# suite.toml [sandbox_daytona].harbor_version, in its own venv on the box.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_HOME="${BENCH_HOME:-$HOME/benchsuite}"
export PATH="$HOME/.local/bin:$PATH"
VER=$(python3 -c 'import tomllib; print(tomllib.load(open("'"$HERE"'/suite.toml","rb"))["sandbox_daytona"]["harbor_version"])')
mkdir -p "$BENCH_HOME"
[ -x "$BENCH_HOME/harborenv/bin/python" ] || uv venv -q -p 3.12 "$BENCH_HOME/harborenv"
VIRTUAL_ENV="$BENCH_HOME/harborenv" uv pip install -q "harbor[daytona]==$VER"
"$BENCH_HOME/harborenv/bin/harbor" --version
