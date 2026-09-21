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
# Model cache placement (2026-09-05). /root on Lium pods is a gocryptfs FUSE
# volume: measured on the eval box, writes 231 MB/s, reads 333 MB/s — the
# whole 72 GB checkpoint pipeline was capped by it (download >= 6 min even
# at line rate; two vLLM engines reading weights => 465-589 s loads). The
# pod's plain local disk (/) did 3.0 GB/s write / 7.4 GB/s read. Use it when
# it has room for a king + challenger + one prefetch with headroom; the
# encrypted volume stays the fallback. Nothing there survives a pod
# recreate, which is fine: every checkpoint is re-fetchable from R2/HF.
if [ -z "${HF_HOME:-}" ]; then
  root_avail_gb=$(df -BG --output=avail / 2>/dev/null | tail -1 | tr -dc '0-9')
  if [ "${root_avail_gb:-0}" -ge 400 ]; then
    export HF_HOME=/hf
  else
    export HF_HOME=/root/hf
  fi
fi
export HF_HOME
echo "[bootstrap] HF_HOME=$HF_HOME ($(df -h --output=avail "$(dirname "$HF_HOME")" 2>/dev/null | tail -1 | tr -d ' ') free)"
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
# Pre-warmed uv download cache (2026-09-21). A cold bootstrap pulls ~11 GB of
# wheels (vLLM/torch cu130, flashinfer cubin + jit-cache) from PyPI/GitHub;
# on a slow Lium pipe that took 24 min + 13 min for the cubin alone and ran
# into the provisioner's 60 min deadline. The cache tarball is published by
# ops/eval_wheelhouse/publish_uv_cache.sh from a healthy pod to the public
# affine-data bucket, keyed by python version + pyproject hash; restoring it
# makes every `uv pip install` below a cache hit. Any failure here falls
# through to the normal pip path.
export UV_CACHE_DIR=${UV_CACHE_DIR:-/root/.cache/uv}
WHEEL_KEY="py312-$(sha256sum /root/affine/pyproject.toml | cut -c1-8)"
WHEEL_URL="${AFFINE_WHEELHOUSE_BASE:-https://data.affine.io/wheelhouse}/uv-cache-${WHEEL_KEY}.tar.zst"
if [ ! -f "$UV_CACHE_DIR/.affine-wheelhouse" ]; then
  t_wh=$(date +%s)
  mkdir -p "$UV_CACHE_DIR"
  if curl -sfIL --max-time 20 "$WHEEL_URL" >/dev/null 2>&1; then
    echo "[bootstrap] restoring uv cache from $WHEEL_URL"
    if command -v zstd >/dev/null 2>&1 || apt-get install -y -qq zstd >/dev/null 2>&1; then
      if curl -sfL --retry 5 --retry-all-errors --max-time 1800 "$WHEEL_URL" \
          | tar -I zstd -xf - -C "$UV_CACHE_DIR" 2>/root/logs/wheelhouse.err; then
        echo "$WHEEL_KEY" > "$UV_CACHE_DIR/.affine-wheelhouse"
        echo "[bootstrap] uv cache restored in $(( $(date +%s) - t_wh ))s"
      else
        echo "[bootstrap] uv cache restore FAILED ($(tail -c 200 /root/logs/wheelhouse.err)); falling back to pip"
      fi
    else
      echo "[bootstrap] zstd unavailable; falling back to pip"
    fi
  else
    echo "[bootstrap] no wheelhouse for $WHEEL_KEY at $WHEEL_URL; pip from the index"
  fi
fi
# Fail closed on install errors (do not hide behind `| tail`).
t_pip=$(date +%s)
uv pip install -e ".[eval]" 2>&1 | tee /root/logs/pip_eval.log | tail -20
echo "[bootstrap] pip [eval] took $(( $(date +%s) - t_pip ))s"
# Prebuilt flashinfer kernels (2026-08-27, Qwen3.8-27B teacher). vLLM >= 0.28
# imports flashinfer unconditionally for GDN models, and bare flashinfer-python
# JIT-compiles at startup — which needs nvcc and dies on pods without a CUDA
# toolkit ("Could not find nvcc"). The cubin + jit-cache wheels ship the
# kernels prebuilt (no nvcc); both live on the flashinfer index, not PyPI.
# Match cubin/jit-cache to whatever flashinfer-python vLLM pulled (0.28 →
# 0.6.16.post3; 0.29 → 0.6.18). A mismatch aborts TP>1 workers with
# "flashinfer-cubin version does not match flashinfer version" before CUDA
# init (seen 2026-09-11 on lunar-raven-18 / vLLM 0.29.0). cuXXX must match
# torch.version.cuda (cu130 for current vLLM torch wheels).
FI_VER=$(python - <<'PY'
import importlib.metadata as m
print(m.version("flashinfer-python"))
PY
)
echo "[bootstrap] flashinfer-python=$FI_VER — installing matching cubin/jit-cache"
uv pip install "flashinfer-cubin==${FI_VER}" \
  --index-url https://flashinfer.ai/whl 2>&1 | tee -a /root/logs/pip_eval.log | tail -5
CUDA_TAG=$(python - <<'PY'
import torch
v = (torch.version.cuda or "13.0").split(".")
print(f"cu{v[0]}{v[1]}")
PY
)
uv pip install "flashinfer-jit-cache==${FI_VER}" \
  --index-url "https://flashinfer.ai/whl/${CUDA_TAG}" 2>&1 | tee -a /root/logs/pip_eval.log | tail -5
python - <<'PY'
import importlib.metadata as m
fi = m.version("flashinfer-python")
cubin = m.version("flashinfer-cubin")
if cubin != fi:
    raise SystemExit(
        f"[bootstrap] FATAL: flashinfer-cubin={cubin} != flashinfer-python={fi}"
    )
print(f"[bootstrap] flashinfer match OK cubin={cubin}")
PY
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
  # Optional pod-local overrides (AFFINE_CHAT_MAX_MODEL_LEN, public key,
  # limits) pushed by ops/king-chat/chatbox.sh; the provisioner never
  # writes this file, so a fresh rental runs on the [chat] toml defaults
  # until the chatbox watcher re-pushes it.
  if [ -f /root/affine/.chat_env ]; then
    set -a
    # shellcheck disable=SC1091
    source /root/affine/.chat_env
    set +a
    echo "[bootstrap] chat overrides loaded from .chat_env"
  fi
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
