#!/bin/bash
# Datagen-machine bootstrap (CPU-only box). Run from /root/affine (repo tar
# or rsync — verify *.py made it, some cloud rsync paths drop them).
# Idempotent: safe to re-run; skips completed steps. Ends by supervising the
# datagen loop in a restart loop.
set -euo pipefail

cd /root/affine
# Secrets (provider keys ENGY / OPENROUTER / CHUTES, HF_TOKEN, DATAGEN_*
# overrides) are written to this 0600 file by the operator over stdin —
# never passed on a command line.
if [ -f /root/affine/.datagen_env ]; then
  # shellcheck disable=SC1091
  source /root/affine/.datagen_env
fi
export HF_HOME=${HF_HOME:-/root/hf}
export DATAGEN_DATA_DIR=${DATAGEN_DATA_DIR:-/root/datagen}
mkdir -p /root/logs "$DATAGEN_DATA_DIR" "$HF_HOME"

echo "[bootstrap] $(date -u) starting (datagen)"

# 1. Docker (agent rollouts and the swebench harness both run in containers).
if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sh
fi
if ! docker info >/dev/null 2>&1; then
  systemctl start docker 2>/dev/null || service docker start 2>/dev/null || true
  sleep 3
  docker info >/dev/null 2>&1 || {
    echo "[bootstrap] FATAL: docker daemon is not running"; exit 1; }
fi

# 2. Python env (uv).
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
if [ ! -d /root/venv ]; then
  uv venv /root/venv --python 3.12
fi
source /root/venv/bin/activate
# Fail closed on install errors (do not hide behind `| tail`).
uv pip install -e ".[datagen]" 2>&1 | tee /root/logs/pip_datagen.log | tail -20
uv pip install "swebench @ git+https://github.com/SWE-rebench/SWE-bench-fork" 2>&1 \
  | tee -a /root/logs/pip_datagen.log | tail -20
python - <<'PY'
import pathlib
import datagen, minisweagent, swebench, datasets
from datagen.config import load_config
n_py = sum(1 for _ in pathlib.Path("/root/affine/datagen").glob("*.py"))
if n_py < 6:
    raise SystemExit(
        f"[bootstrap] FATAL: only {n_py} .py sources under datagen/ "
        "(cloud rsync has been observed to drop *.py — upload a tar instead)"
    )
from datagen.providers import load_providers
cfg = load_config()
names = [p.name for p in load_providers()]
print(f"[bootstrap] IMPORT_OK providers={names} datasets={cfg.datasets} "
      f"hf_repo={cfg.hf_repo} n_py={n_py}")
PY

# 3. Supervise the loop.
#    `|| code=$?` is load-bearing: under `set -e` a bare nonzero exit of the
#    loop (crash, SIGTERM, OOM) aborts this whole script and the restart loop
#    never runs — exactly the always-online failure mode this loop prevents.
while true; do
  # A loop that died mid-batch can leave an orphaned agent run behind (its
  # own process group survives the parent); kill it before relaunching so
  # two agent phases never run concurrently. Bracket trick keeps pgrep from
  # matching this script's own command line.
  pkill -f '[m]ini-extra swebench' 2>/dev/null || true
  pkill -f '[s]webench.harness.run_evaluation' 2>/dev/null || true
  # Re-source per launch: operators tune DATAGEN_* in the env file and bounce
  # only the loop — without this the supervisor keeps serving the env it was
  # born with (bit us live: a provider-pool change was silently ignored).
  if [ -f /root/affine/.datagen_env ]; then
    # shellcheck disable=SC1091
    source /root/affine/.datagen_env
  fi
  echo "[bootstrap] $(date -u) launching datagen loop"
  code=0
  python -m datagen.loop >> /root/logs/datagen.log 2>&1 || code=$?
  echo "[bootstrap] $(date -u) datagen loop exited code=$code; restarting in 30s"
  sleep 30
done
