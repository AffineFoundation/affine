#!/bin/bash
# Meta ARE (are-benchmark) for the Gaia2 cell (gaia2_cell.py), pinned by suite.toml
# [[envs]] gaia2-ambiguity.runner.version, in its own venv on the box, plus a local mirror
# of the scenarios' file system (gaia2_filesystem/demo_filesystem) so ARE does not hit the
# HF API per file (1000 req / 5 min cap).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_HOME="${BENCH_HOME:-$HOME/benchsuite}"
export PATH="$HOME/.local/bin:$PATH"
VER=$(python3 -c 'import tomllib; d=tomllib.load(open("'"$HERE"'/suite.toml","rb")); print([e for e in d["envs"] if e["id"]=="gaia2-ambiguity"][0]["runner"]["version"])')
mkdir -p "$BENCH_HOME"
[ -x "$BENCH_HOME/areenv/bin/python" ] || uv venv -q -p 3.12 "$BENCH_HOME/areenv"
VIRTUAL_ENV="$BENCH_HOME/areenv" uv pip install -q "meta-agents-research-environments==$VER"
if [ ! -d "$BENCH_HOME/gaia2_fs/demo_filesystem" ]; then
  "$BENCH_HOME/areenv/bin/python" - "$BENCH_HOME/gaia2_fs" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download("meta-agents-research-environments/gaia2_filesystem", repo_type="dataset",
                  allow_patterns=["demo_filesystem/*", "demo_filesystem/**"], local_dir=sys.argv[1], max_workers=8)
PY
fi
echo "are $("$BENCH_HOME/areenv/bin/python" -c 'import importlib.metadata as m; print(m.version("meta-agents-research-environments"))'), fs files $(find "$BENCH_HOME/gaia2_fs/demo_filesystem" -type f | wc -l)"
