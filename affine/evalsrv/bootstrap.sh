#!/bin/bash
# Eval-machine bootstrap. Run from /root/affine (rsynced by the provisioner).
# Idempotent: safe to re-run; skips completed steps. Ends by supervising the
# eval server in a restart loop (fatal CUDA errors self-kill; we relaunch).
set -euo pipefail

cd /root/affine
# Secrets (HF_TOKEN, AFFINE_EVAL_TOKEN, ...) are written to this 0600 file by
# the provisioner over stdin — never passed on a command line.
if [ -f /root/affine/.eval_env ]; then
  # shellcheck disable=SC1091
  source /root/affine/.eval_env
fi
export HF_HOME=${HF_HOME:-/root/hf}
# Rust multi-stream downloads (hf_transfer, installed via the [eval] extra).
# Model pulls are the fixed per-duel cost; this takes them to near line rate.
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}
export AFFINE_DATA_DIR=${AFFINE_DATA_DIR:-/root/affine_data}
export AFFINE_EVAL_PORT=${AFFINE_EVAL_PORT:-9000}
mkdir -p /root/logs "$AFFINE_DATA_DIR" "$HF_HOME"

echo "[bootstrap] $(date -u) starting"

# 1. Python env (uv).
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
if [ ! -d /root/venv ]; then
  uv venv /root/venv --python 3.12
fi
source /root/venv/bin/activate
# Fail closed on install errors (do not hide behind `| tail`).
uv pip install -e ".[eval]" 2>&1 | tee /root/logs/pip_eval.log | tail -20
# Prebuilt flashinfer kernels (2026-08-27, Qwen3.8-27B teacher). vLLM >= 0.28
# imports flashinfer unconditionally for GDN models, and bare flashinfer-python
# JIT-compiles at startup — which needs nvcc and dies on pods without a CUDA
# toolkit ("Could not find nvcc"). The cubin + jit-cache wheels ship the
# kernels prebuilt (no nvcc); both live on the flashinfer index, not PyPI.
# Pin to vLLM 0.28.0's required flashinfer-python==0.6.16.post3; the cuXXX
# suffix must match torch.version.cuda (cu130 for the torch vLLM 0.28 pulls).
uv pip install "flashinfer-cubin==0.6.16.post3" \
  --index-url https://flashinfer.ai/whl 2>&1 | tail -3
uv pip install "flashinfer-jit-cache==0.6.16.post3" \
  --index-url https://flashinfer.ai/whl/cu130 2>&1 | tail -3
python - <<'PY'
import affine, evalsrv
from affine.config import load_config
n_py = sum(1 for _ in __import__("pathlib").Path("/root/affine/affine").glob("*.py"))
n_py += sum(1 for _ in __import__("pathlib").Path("/root/affine/evalsrv").glob("*.py"))
if n_py < 10:
    raise SystemExit(
        f"[bootstrap] FATAL: only {n_py} .py sources under affine/evalsrv "
        "(lium rsync has been observed to drop *.py — upload a tar instead)"
    )
print("[bootstrap] IMPORT_OK", load_config().dataset.manifest_key, f"n_py={n_py}")
PY

ROLE=${AFFINE_ROLE:-duel}
echo "[bootstrap] AFFINE_ROLE=$ROLE"

if [ "$ROLE" = "bench" ]; then
  # Dedicated SWE bench pod: agent + docker harness deps, no turn corpus.
  uv pip install "mini-swe-agent" "datasets" "pyarrow" 2>&1 \
    | tee /root/logs/pip_bench.log | tail -20
  uv pip install "swebench @ git+https://github.com/SWE-rebench/SWE-bench-fork" 2>&1 \
    | tee -a /root/logs/pip_bench.log | tail -20
  mkdir -p /root/bench
elif [ "$ROLE" = "chat" ]; then
  # Public king-chat pod: serves the current king only — no corpus, no
  # bench deps. chatsrv polls the public snapshot for king changes itself.
  :
else
  # 2. Turn corpus D: manifest + shard sync from the public bucket. Fail-closed:
  #    the sync verifies the manifest hash against its immutable published copy
  #    and every shard sha256, and exits nonzero without a verified corpus.
  python -m evalsrv.corpus --sync
fi

# 3. Supervise the eval server. On duel role, teacher warmup happens inside
#    the app and /health flips ok=true when it is servable. On bench role,
#    /health is ok immediately (no teacher).
#    `|| code=$?` is load-bearing: under `set -e` a bare nonzero exit of the
#    server (crash, SIGTERM, CUDA self-kill) aborts this whole script and the
#    restart loop never runs — exactly the always-online failure mode this
#    loop exists to prevent.
SERVER_MODULE=evalsrv.server
if [ "$ROLE" = "chat" ]; then
  SERVER_MODULE=evalsrv.chatsrv
fi
while true; do
  echo "[bootstrap] $(date -u) launching $SERVER_MODULE role=$ROLE"
  code=0
  python -m "$SERVER_MODULE" >> /root/logs/evalsrv.log 2>&1 || code=$?
  echo "[bootstrap] $(date -u) $SERVER_MODULE exited code=$code; restarting in 10s"
  sleep 10
done
