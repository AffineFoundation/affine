#!/usr/bin/env bash
# Bench env for the eval box (mini-swe-agent + SWE-rebench fork harness),
# same stack Track A/B used. Also pre-pulls the proxy panel docker images.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
uv venv /root/benchenv --python 3.12
VIRTUAL_ENV=/root/benchenv uv pip install -q mini-swe-agent datasets pyarrow pyyaml requests
VIRTUAL_ENV=/root/benchenv uv pip install -q "swebench @ git+https://github.com/SWE-rebench/SWE-bench-fork"
/root/benchenv/bin/python -c "import swebench, minisweagent, yaml; print('bench env ok')"
xargs -P 4 -n 1 docker pull < /root/work/panels/images_proxy.txt > /root/work/prepull_proxy.log 2>&1
echo BENCHENV_DONE
