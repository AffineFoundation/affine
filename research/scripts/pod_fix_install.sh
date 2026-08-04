#!/bin/bash
set -euxo pipefail
cd /root/affine
export PATH="$HOME/.local/bin:$PATH"
source /root/venv/bin/activate
echo "=== tree ==="
ls -la
ls affine | head
ls evalsrv | head
echo "=== show ==="
uv pip show affine || true
echo "=== reinstall editable (no extras first) ==="
uv pip install -e . 2>&1 | tee /root/logs/pip_affine.log | tail -50
echo "=== import ==="
python - <<'PY'
import affine, evalsrv
from affine.config import load_config
print("affine", affine.__file__)
print("evalsrv", evalsrv.__file__)
print("repo", load_config().dataset.turns_hf_repo)
print("IMPORT_OK")
PY
echo "=== ensure eval extras ==="
# Skip tau2 git dep for duel smoke — install fastapi/uvicorn/vllm/transformers only.
uv pip install fastapi uvicorn transformers 'vllm>=0.8' 2>&1 | tee /root/logs/pip_extras.log | tail -30
python -c 'import fastapi, vllm; print("EXTRAS_OK", fastapi.__version__)'
