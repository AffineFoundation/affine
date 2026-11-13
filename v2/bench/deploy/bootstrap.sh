#!/bin/bash
# Stand up an Affine bench node from this repo (pod side). Idempotent: re-running only installs what is missing.
#   git clone https://github.com/AffineFoundation/affine.git /data/bench/repo && bash /data/bench/repo/v2/bench/deploy/bootstrap.sh
# R below is <clone>/v2, so every bench/... path in the lifted scripts stays as it was.
# Layout (plain disk only, never the gocryptfs /root): /data/bench/{repo,venv-vllm,venv-vf,models,hf,runs,logs}
# No secrets are ever placed on the node: models come from public HF or from short-lived presigned GETs piped in by the host.
set -euo pipefail
B=/data/bench; R=$(cd "$(dirname "$0")/../.." && pwd); D=$R/bench/deploy
mkdir -p $B/{models,hf,runs,logs,tmp,uvcache}
cat > $B/env.sh <<E
export HOME=$B/home XDG_CACHE_HOME=$B/cache HF_HOME=$B/hf UV_CACHE_DIR=$B/uvcache TMPDIR=$B/tmp
export VLLM_CACHE_ROOT=$B/cache/vllm TRITON_CACHE_DIR=$B/cache/triton TORCHINDUCTOR_CACHE_DIR=$B/cache/inductor
export HF_HUB_ENABLE_HF_TRANSFER=0 HF_XET_HIGH_PERFORMANCE=1 TOKENIZERS_PARALLELISM=false
export BENCH_ROOT=$B BENCH_REPO=$R PATH=$B/venv-vf/bin:/usr/local/bin:\$PATH
E
mkdir -p $B/home $B/cache; . $B/env.sh
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/0.9.5/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh; }
apt-get -qq install -y jq >/dev/null 2>&1 || true
# 1) inference: vLLM pinned to the exact closure the SN120 probe/perf-eval pods run (requirements-vllm.txt)
if ! $B/venv-vllm/bin/python -c "import vllm" 2>/dev/null; then
  uv venv -q -p 3.12 $B/venv-vllm
  VIRTUAL_ENV=$B/venv-vllm uv pip install -q -r $D/requirements-vllm.txt
fi
# 2) harness: verifiers (v1) + prime CLI + pinned Hub envs (wheel URL + sha256) + our own envs from bench/environments
if ! $B/venv-vf/bin/python -c "import verifiers, math500" 2>/dev/null || [ "${BENCH_REINSTALL_VF:-0}" = 1 ]; then
  uv venv -q -p 3.12 $B/venv-vf
  VIRTUAL_ENV=$B/venv-vf uv pip install -q -r $D/requirements-harness.txt
fi
VIRTUAL_ENV=$B/venv-vf uv pip install -q -r $D/requirements-envs.lock --no-deps
for e in $R/bench/environments/*/; do [ -f "$e/pyproject.toml" ] || continue; VIRTUAL_ENV=$B/venv-vf uv pip install -q --no-deps -e "$e"; done
VIRTUAL_ENV=$B/venv-vf uv pip install -q --no-deps -e $R/bench
$B/venv-vllm/bin/python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'gpus', torch.cuda.device_count())"
$B/venv-vf/bin/python -c "import verifiers; print('verifiers', verifiers.__version__)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sort | uniq -c
df -h $B | tail -1
echo BOOTSTRAP_OK
