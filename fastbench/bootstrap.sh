#!/bin/bash
# fastbench pod bootstrap. Run from /root/fastbench (uploaded by ops/rent.py).
# Idempotent; ends by supervising the server in a restart loop.
set -euo pipefail

cd /root/fastbench

# Single supervisor only. Redeploys that miss the old loop otherwise leave two
# supervisors racing for the port: one wins, the other crash-loops, and which
# process owns a job becomes unknowable.
exec 9>/var/lock/fastbench-bootstrap.lock
if ! flock -n 9; then
  echo "[bootstrap] another supervisor holds the lock; exiting" >&2
  exit 0
fi

if [ -f /root/fastbench/.env ]; then
  # set -a: exported, so child processes (server, vllm, agents) see them.
  set -a
  # shellcheck disable=SC1091
  source /root/fastbench/.env
  set +a
fi
export HF_HOME=${HF_HOME:-/root/hf}
# Fast parallel model downloads. hf_transfer is retired in current
# huggingface_hub; Xet high-performance mode is its replacement.
export HF_XET_HIGH_PERFORMANCE=${HF_XET_HIGH_PERFORMANCE:-1}
export FASTBENCH_DATA_DIR=${FASTBENCH_DATA_DIR:-/root/fastbench_data}
export FASTBENCH_PORT=${FASTBENCH_PORT:-9100}
mkdir -p /root/logs "$FASTBENCH_DATA_DIR" "$HF_HOME"

echo "[bootstrap] $(date -u) starting"

# 0. Docker must work (DinD template) — rollouts and evals depend on it.
if ! docker info >/dev/null 2>&1; then
  echo "[bootstrap] FATAL: docker is not usable on this pod" >&2
  exit 1
fi

# 1. Python env (uv).
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
if [ ! -d /root/venv ]; then
  uv venv /root/venv --python 3.12
fi
source /root/venv/bin/activate
uv pip install -e ".[serve]" 2>&1 | tee /root/logs/pip_fastbench.log | tail -5
# SWE-rebench harness fork (same stack the evalsrv bench pods run).
uv pip install "swebench @ git+https://github.com/SWE-rebench/SWE-bench-fork" \
  2>&1 | tee -a /root/logs/pip_fastbench.log | tail -5

# 1b. Align the CUDA compiler with the CUDA runtime headers.
# FlashInfer's bundled cccl refuses to build unless nvcc's version EXACTLY
# equals CUDART_VERSION, and the vLLM wheel set ships nvcc 13.3 against
# runtime 13.0. Without this every JIT kernel (attention fmha_gen, fused MoE,
# sampling) fails with "CUDA compiler and CUDA toolkit headers are
# incompatible" and the engine never starts.
CU_VER=$(python - <<'PY'
import pathlib, re, sysconfig
hdr = (pathlib.Path(sysconfig.get_paths()["purelib"]) / "nvidia" / "cu13"
       / "include" / "cuda_runtime_api.h")
m = re.search(r"define CUDART_VERSION\s+(\d+)", hdr.read_text()) if hdr.exists() else None
v = int(m.group(1)) if m else 0
print(f"{v // 1000}.{v % 1000 // 10}" if v else "")
PY
)
NVCC_BIN="$(python -c 'import sysconfig,pathlib;print(pathlib.Path(sysconfig.get_paths()["purelib"])/"nvidia"/"cu13"/"bin"/"nvcc")')"
NVCC_VER=$("$NVCC_BIN" --version 2>/dev/null | grep -oE 'release [0-9]+\.[0-9]+' | cut -d' ' -f2 || true)
if [ -n "$CU_VER" ] && [ "$NVCC_VER" != "$CU_VER" ]; then
  echo "[bootstrap] pinning CUDA compiler ${NVCC_VER:-none} -> runtime $CU_VER"
  # cicc lives in nvidia-nvvm, ptxas in nvidia-cuda-nvcc: a partial downgrade
  # leaves a PTX-version mismatch, so all five move together.
  uv pip install "nvidia-cuda-nvcc==${CU_VER}.*" "nvidia-cuda-crt==${CU_VER}.*" \
    "nvidia-cuda-cccl==${CU_VER}.*" "nvidia-nvvm==${CU_VER}.*" \
    "nvidia-nvjitlink==${CU_VER}.*" 2>&1 | tail -3
fi

# 1c. The pip CUDA wheels ship only versioned sonames (libcudart.so.13), but
# the FlashInfer JIT links with -lcudart, which needs the unversioned name.
python - <<'PY'
import pathlib, sysconfig
lib = (pathlib.Path(sysconfig.get_paths()["purelib"]) / "nvidia" / "cu13"
       / "lib")
for so in sorted(lib.glob("*.so.*")) if lib.is_dir() else []:
    link = lib / (so.name.split(".so.")[0] + ".so")
    if not link.exists():
        link.symlink_to(so.name)
        print(f"[bootstrap] linked {link.name} -> {so.name}")
PY

python - <<'PY'
import fastbench, minisweagent, swebench, vllm  # noqa: F401
from fastbench.pool import load_pool
print("[bootstrap] IMPORT_OK pool", load_pool()["n_instances"])
PY

# 2. Warm the HF dataset cache (harness + subset prep both read it).
python - <<'PY'
from datasets import load_dataset
from fastbench.pool import load_pool
pool = load_pool()
ds = load_dataset(pool["dataset"], split=pool.get("split", "test"))
print(f"[bootstrap] dataset cached: {len(ds)} rows")
PY

# 3. Bake pool images in the background (idempotent, ~1h cold, no-op warm).
if [ "${FASTBENCH_SKIP_PREPULL:-0}" != "1" ]; then
  (nohup python scripts/prepull_images.py --workers 12 \
     </dev/null >> /root/logs/prepull.log 2>&1 &)
  echo "[bootstrap] pool image prepull running in background (logs/prepull.log)"
fi

# 4. Supervise the server (`|| code=$?` keeps the loop alive under set -e).
while true; do
  echo "[bootstrap] $(date -u) launching fastbench.server"
  code=0
  python -m fastbench.server >> /root/logs/fastbench.log 2>&1 || code=$?
  echo "[bootstrap] $(date -u) server exited code=$code; restarting in 10s"
  sleep 10
done
