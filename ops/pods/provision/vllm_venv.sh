#!/bin/bash
# Pod side, idempotent: the SN120 vLLM 0.32 closure (bench/deploy/requirements-vllm.txt, the exact probe pins) + math-verify 0.9.0
# in ROOT/VENV, plain disk only (never the gocryptfs /root). Same recipe as the live perf-eval and prior pods.
#   bash vllm_venv.sh ROOT [VENV_NAME=venv]        (run from a clone of github.com/AffineFoundation/affine)
#   CHECK=1 bash vllm_venv.sh ROOT [VENV_NAME]     -> no installs; prints pin drift vs the requirements and exits 0/1
set -euo pipefail
ROOT=$1; V=$ROOT/${2:-venv}; REPO=$(cd "$(dirname "$0")/../../.." && pwd); REQ=$REPO/bench/deploy/requirements-vllm.txt
EXTRA="math-verify==0.9.0 latex2sympy2_extended==1.11.0 antlr4-python3-runtime==4.13.2"
mkdir -p $ROOT/tmp $ROOT/uvcache $ROOT/home $ROOT/cache
export UV_CACHE_DIR=$ROOT/uvcache TMPDIR=$ROOT/tmp UV_PYTHON_PREFERENCE=only-system
command -v uv >/dev/null || { [ "${CHECK:-0}" = 1 ] && { echo "uv missing"; exit 1; }; curl -LsSf https://astral.sh/uv/0.9.5/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh; }
want() { { grep -v -E '^\s*(#|$)' $REQ; for x in $EXTRA; do echo $x; done; } | sed 's/_/-/g' | tr 'A-Z' 'a-z' | sort -u; }
have() { uv pip freeze -q --python $V/bin/python 2>/dev/null | grep -v -E '^\s*(#|$|-e )' | sed 's/_/-/g' | tr 'A-Z' 'a-z' | sort -u; }
if [ "${CHECK:-0}" = 1 ]; then
  [ -x $V/bin/python ] || { echo "MISSING venv $V"; exit 1; }
  miss=$(comm -23 <(want) <(have)); n=$(echo -n "$miss" | grep -c . || true)
  echo "venv $V: $(have | wc -l) pkgs; required pins missing/different: $n"; [ $n = 0 ] || echo "$miss" | head -20
  [ $n = 0 ] && echo VENV_MATCH || { echo VENV_DRIFT; exit 1; }
  exit 0
fi
if [ -x $V/bin/python ] && [ -z "$(comm -23 <(want) <(have))" ]; then echo "venv $V already matches"; else
  [ -x $V/bin/python ] || uv venv -q -p 3.12 $V
  VIRTUAL_ENV=$V uv pip install -q -r $REQ
  VIRTUAL_ENV=$V uv pip install -q $EXTRA
fi
$V/bin/python -c "import vllm, torch, math_verify; print('vllm', vllm.__version__, 'torch', torch.__version__, 'gpus', torch.cuda.device_count())"
echo VENV_OK
