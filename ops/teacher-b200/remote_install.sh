#!/usr/bin/env bash
# Run on the teacher pod after SCP. Prefetches GLM-Air and launches serve.
set -euo pipefail

REPO=${TEACHER_REPO:-zai-org/GLM-4.5-Air-FP8}
PORT=${TEACHER_PORT:-40000}
VLLM_VER=${VLLM_VER:-0.22.1}
HF_TOKEN=${HF_TOKEN:?HF_TOKEN required}

export HF_HOME=/root/hf
export HUGGINGFACE_HUB_CACHE=/root/hf/hub
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export HF_TOKEN

mkdir -p /root/hf/hub /root/logs /root/teacher

if [[ ! -x /root/venv/bin/python || ! -x /root/venv/bin/pip ]]; then
  rm -rf /root/venv
  python3 -m venv /root/venv
fi
# shellcheck disable=SC1091
source /root/venv/bin/activate
# Avoid in-place pip self-upgrade (can delete /root/venv/bin/pip mid-install).
python -m pip install -U wheel
python -m pip install "vllm==${VLLM_VER}" huggingface_hub

TEACHER_REPO="$REPO" python - <<'PY'
import os
from huggingface_hub import snapshot_download
repo = os.environ["TEACHER_REPO"]
print("downloading", repo, flush=True)
path = snapshot_download(repo, token=os.environ["HF_TOKEN"])
print("OK", path, flush=True)
PY

chmod +x /root/teacher/serve_teacher.sh

# Stop prior teacher engines only (narrow pattern).
pids=$(pgrep -f 'vllm serve .*GLM-4\.5-Air' || true)
if [[ -n "${pids:-}" ]]; then
  # shellcheck disable=SC2086
  kill $pids 2>/dev/null || true
  sleep 3
fi

nohup /root/teacher/serve_teacher.sh >/root/logs/vllm_teacher.nohup 2>&1 &
echo $! >/root/logs/vllm_teacher.pid
echo "launched pid=$(cat /root/logs/vllm_teacher.pid)"

for i in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/tmp/teacher_models.json; then
    echo "READY after ~$((i * 10))s"
    head -c 500 /tmp/teacher_models.json; echo
    exit 0
  fi
  # Surface early crashes
  if [[ -f /root/logs/vllm_teacher.pid ]]; then
    pid=$(cat /root/logs/vllm_teacher.pid)
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "vLLM died early; log tail:"
      tail -n 100 /root/logs/vllm_teacher.log || true
      exit 1
    fi
  fi
  sleep 10
done
echo "TIMEOUT waiting for :${PORT}"
tail -n 100 /root/logs/vllm_teacher.log || true
exit 1
