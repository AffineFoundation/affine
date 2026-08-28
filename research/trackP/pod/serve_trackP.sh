#!/usr/bin/env bash
# Track P vLLM launcher (H200): official container, default v0.10.1.1
# (v0.27.1/:latest is broken on H200 under load -- engine dies silently).
# usage: serve_trackP.sh <repo_or_path> <port> <gpus_csv> [extra vllm args...]
set -euo pipefail
REPO="${1:?repo}"; PORT="${2:?port}"; GPUS="${3:?gpus}"; shift 3 || true
NGPU="$(echo "$GPUS" | tr ',' '\n' | grep -c .)"
NAME="vllm_${PORT}"
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.10.1.1}"
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --ipc=host \
  --gpus "\"device=${GPUS}\"" \
  -v /dshare:/dshare -v /dshare/hf:/root/hf \
  -e HF_HOME=/root/hf \
  -e HF_TOKEN="$(cat /root/.hf_token)" \
  -e VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 \
  -p "127.0.0.1:${PORT}:8000" \
  "$IMAGE" \
  --model "$REPO" \
  --served-model-name "$REPO" \
  --tensor-parallel-size "$NGPU" \
  --gpu-memory-utilization 0.90 \
  "$@"
echo "container $NAME serving $REPO gpus=$GPUS tp=$NGPU -> 127.0.0.1:$PORT"
