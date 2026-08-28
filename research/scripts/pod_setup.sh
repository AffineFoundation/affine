#!/usr/bin/env bash
# One-shot setup for a fresh Lium Blackwell pod (B200/B300).
#
# These pods ship the NVIDIA driver and a torch wheel but no CUDA toolkit, so
# anything that JIT-compiles kernels (flashinfer, which vLLM hard-requires on
# Blackwell) fails to build. Three things must all be true for the build to
# work, and each one cost a debugging round on the first pod:
#
#   1. nvcc must exist and be findable. It ships inside the nvidia cu13 wheels
#      that vLLM pulls in, so CUDA_HOME points there rather than /usr/local/cuda.
#   2. nvcc's version must match the CUDA runtime headers. The wheels install
#      mismatched versions by default (nvcc 13.3 against 13.0 headers), which
#      fails with "CUDA compiler and CUDA toolkit headers are incompatible".
#   3. ninja must be on PATH. It lives in the venv's bin, which is skipped when
#      vllm is invoked by absolute path.
#
# vLLM lives in its own venv so installing it cannot disturb the torch version
# that the training scripts use.
set -euo pipefail

VENV=/root/vllmenv
WORK=/root/work
mkdir -p "$WORK" /root/bench

log() { echo "[setup] $*"; }

log "installing uv"
command -v uv >/dev/null 2>&1 || python3 -m pip install -q --break-system-packages uv

log "creating vllm venv"
uv venv "$VENV" --python 3.12

log "installing vllm (large; several minutes)"
VIRTUAL_ENV="$VENV" uv pip install -q vllm ninja

# --- pin nvcc to the runtime header version -------------------------------
SP="$VENV/lib/python3.12/site-packages"
RT_VER="$(VIRTUAL_ENV=$VENV uv pip list 2>/dev/null \
          | awk '/^nvidia-cuda-runtime /{print $2}' | cut -d. -f1,2)"
log "cuda runtime headers report version: ${RT_VER:-unknown}"
if [ -n "$RT_VER" ]; then
  log "pinning nvcc + crt to ${RT_VER}.* so the header check passes"
  VIRTUAL_ENV="$VENV" uv pip install -q \
    "nvidia-cuda-nvcc==${RT_VER}.*" "nvidia-cuda-crt==${RT_VER}.*" || \
    log "WARN: could not pin nvcc; JIT builds may fail"
fi

CUDA_HOME="$SP/nvidia/cu13"
if [ -x "$CUDA_HOME/bin/nvcc" ]; then
  log "nvcc: $("$CUDA_HOME/bin/nvcc" --version | grep -o 'release [0-9.]*')"
else
  log "WARN: no nvcc at $CUDA_HOME/bin/nvcc"
fi

# --- bench dependencies (system python, matching evalsrv/bootstrap.sh) ----
log "installing bench deps"
python3 -m pip install -q --break-system-packages \
  mini-swe-agent datasets pyarrow requests
python3 -m pip install -q --break-system-packages \
  "swebench @ git+https://github.com/SWE-rebench/SWE-bench-fork"

log "verifying"
"$VENV/bin/python" -c "import vllm; print('  vllm', vllm.__version__)"
python3 -c "import swebench; print('  swebench ok')"
command -v mini-extra >/dev/null && echo "  mini-extra ok"
docker info >/dev/null 2>&1 && echo "  docker ok"
echo "SETUP_DONE"
