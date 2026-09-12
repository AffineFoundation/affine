#!/bin/bash
# Build the eval-driver environment for the benchmark suite in $BENCH_HOME
# (default ~/benchsuite): a pinned `verifiers` checkout with its own uv venv
# and every suite taskset installed EDITABLE + --no-deps (letting uv resolve
# `verifiers>=0.3.1` would replace the editable checkout with a PyPI release —
# the datagen pods learned this the hard way), then the tasksets' own deps.
# Idempotent. Run `uv run eval` from $BENCH_HOME/verifiers with UV_NO_SYNC=1
# (a plain `uv run` re-syncs the venv from the lockfile and drops the tasksets).
#
#   BENCH_HOME=~/benchsuite bash install_eval_env.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_HOME="${BENCH_HOME:-$HOME/benchsuite}"
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install -q 3.12
# tomllib needs Python >= 3.11; the host python may be older (Prime pods ship 3.10).
PY="uv run --no-project --python 3.12 --quiet python"
mkdir -p "$BENCH_HOME"
cd "$BENCH_HOME"

read_toml() { $PY - "$HERE/suite.toml" "$1" <<'PY'
import sys, tomllib
d = tomllib.load(open(sys.argv[1], "rb"))
print(d["suite"][sys.argv[2]])
PY
}
VF_REPO=$(read_toml verifiers_repo); VF_COMMIT=$(read_toml verifiers_commit)
RE_REPO=$(read_toml research_envs_repo); RE_COMMIT=$(read_toml research_envs_commit)

[ -d verifiers ] || git clone -q "$VF_REPO" verifiers
[ -d research-environments ] || git clone -q "$RE_REPO" research-environments
(cd verifiers && git fetch -q origin && git checkout -q "$VF_COMMIT")
(cd research-environments && git fetch -q origin && git checkout -q "$RE_COMMIT")

cd verifiers
uv sync --extra harbor 2>&1 | tail -2
export VIRTUAL_ENV="$BENCH_HOME/verifiers/.venv"

# Tasksets, editable, no deps (keeps the editable verifiers).
mapfile -t INSTALLS < <($PY - "$HERE/suite.toml" <<'PY'
import sys, tomllib
d = tomllib.load(open(sys.argv[1], "rb"))
for e in d["envs"]:
    print(e["install"])
PY
)
for inst in "${INSTALLS[@]}"; do
  if [[ "$inst" == envs/* ]]; then path="$HERE/$inst"; else path="$BENCH_HOME/research-environments/environments/$inst"; fi
  uv pip install -q --no-deps -e "$path"
done
# gpqa-strict wraps upstream gpqa
uv pip install -q --no-deps -e "$BENCH_HOME/research-environments/environments/science/gpqa"

# The tasksets' own dependencies (from their pyproject files), verifiers excluded.
uv pip install -q datasets math-verify langdetect nltk immutabledict spacy emoji syllapy \
  "setuptools<78" pip huggingface-hub python-dateutil filelock "soundfile>=0.13.0" \
  "bfcl-eval @ git+https://github.com/mikasenghaas/gorilla.git@898763a#subdirectory=berkeley-function-call-leaderboard"
uv pip install -q hf_transfer
uv tool install -q prime   # separate tool env: prime depends on PyPI verifiers, which would shadow the checkout
uv pip install -q --no-deps -e .   # re-assert the editable verifiers checkout
.venv/bin/python -c "import nltk; nltk.download('punkt_tab', quiet=True)"

echo "== check"
for ts in aime25 math500 mmlu-pro gpqa-strict ifbench ifeval humaneval livecodebench bfcl-v3 minif2f oolong-synth mrcr-v2 graphwalks swebench-verified; do
  .venv/bin/eval "$ts" --dry-run -n 1 --no-rich --no-push -m x >/dev/null 2>&1 && echo "ok  $ts" || echo "BAD $ts"
done
echo "INSTALL_DONE $(git -C "$BENCH_HOME/verifiers" rev-parse --short HEAD) $(git -C "$BENCH_HOME/research-environments" rev-parse --short HEAD)"
