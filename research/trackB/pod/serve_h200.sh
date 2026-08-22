#!/usr/bin/env bash
# vLLM launcher for H200 using the official container (pip vLLM 0.27.1 died
# in engine warmup: first the flashinfer sampler JIT (no nvcc on the pod),
# then a scatter-gather assert in the native-sampler warmup path; the
# container ships a full toolchain + AOT flashinfer, so neither path breaks).
# usage: serve_h200.sh <repo_or_path> <port> <gpus_csv> [extra vllm args...]
set -euo pipefail
REPO="${1:?repo}"; PORT="${2:?port}"; GPUS="${3:?gpus}"; shift 3 || true
NGPU="$(echo "$GPUS" | tr ',' '\n' | grep -c .)"
NAME="vllm_${PORT}"
# default to v0.11.0: 0.27.1 (:latest) is broken on H200 for Qwen3 — engine
# dies under load and, even when alive, emits degenerate "!!!!" tokens.
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.11.0}"
# --no-async-scheduling fixes a fatal parallel-sampling assert on :latest
# (0.27.1); the flag does not exist on older tags where async is opt-in.
NOASYNC="--no-async-scheduling"
case "$IMAGE" in *v0.1*) NOASYNC="" ;; esac
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --ipc=host \
  --gpus "\"device=${GPUS}\"" \
  -v /dshare:/dshare -v /dshare/hf:/root/hf \
  -e HF_HOME=/root/hf \
  -e HF_TOKEN="$(cat /root/.hf_token)" \
  -e VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 \
  -e CPATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include \
  -e LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib \
  -p "127.0.0.1:${PORT}:8000" \
  "$IMAGE" \
  --model "$REPO" \
  --served-model-name "$REPO" \
  --tensor-parallel-size "$NGPU" \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  $NOASYNC \
  "$@"
echo "container $NAME serving $REPO gpus=$GPUS tp=$NGPU -> 127.0.0.1:$PORT"
