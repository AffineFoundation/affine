#!/usr/bin/env bash
# Serve a model with vLLM using the official prebuilt image.
#
# Why Docker instead of a venv: on these Blackwell pods vLLM's flashinfer
# dependency JIT-compiles kernels at startup, and the pod has no matching CUDA
# toolchain. The generated PTX (9.3) is newer than any ptxas we can install
# alongside the 13.0 headers, so the build cannot be satisfied locally. The
# official image ships those kernels prebuilt, so nothing compiles at startup.
#
# Designed to run several instances at once, one per GPU, for parallel
# benchmarking of different checkpoints: each call gets its own container name,
# port and GPU set.
#
# usage: pod_serve_docker.sh <hf_repo_or_local_path> <port> <gpus_csv> [extra vllm args...]
#   pod_serve_docker.sh Qwen/Qwen3.8-27B 8001 0
#   pod_serve_docker.sh /root/ckpt/round3 8003 2 --max-model-len 16384
set -euo pipefail

REPO="${1:?hf repo or local path}"
PORT="${2:?port}"
GPUS="${3:?gpu ids, e.g. 0 or 0,1}"
shift 3 || true

NGPU="$(echo "$GPUS" | tr ',' '\n' | grep -c .)"
NAME="vllm_$(echo "$REPO" | tr '/.' '__')_$PORT"
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"

# /root on these pods is a gocryptfs FUSE mount, and bind-mounting out of it
# fails with "change mount propagation through procfd". Everything a container
# must see therefore lives under /opt, which is plain overlay/xfs.
HF_DIR="${HF_DIR:-/opt/hf}"

docker rm -f "$NAME" >/dev/null 2>&1 || true

# Local checkpoints must be visible inside the container at the same path.
MOUNTS=(-v "$HF_DIR":"$HF_DIR")
case "$REPO" in
  /*) MOUNTS+=(-v "$(dirname "$REPO")":"$(dirname "$REPO")") ;;
esac
# Adapters written by the adversarial loop are hot-loaded by path, so the
# container has to see that directory too.
CKPT_DIR="${CKPT_DIR:-/opt/ckpt}"
[ -d "$CKPT_DIR" ] && MOUNTS+=(-v "$CKPT_DIR":"$CKPT_DIR")

echo "serving $REPO  port=$PORT  gpus=$GPUS (tp=$NGPU)  container=$NAME"

docker run -d --name "$NAME" \
  --gpus "device=$GPUS" \
  --network host \
  --ipc host \
  --shm-size 32g \
  "${MOUNTS[@]}" \
  -e HF_HOME="$HF_DIR" \
  -e VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
  -e HF_TOKEN="$(cat /root/.hf_token 2>/dev/null || true)" \
  "$IMAGE" \
  --model "$REPO" \
  --served-model-name "$REPO" \
  --port "$PORT" \
  --tensor-parallel-size "$NGPU" \
  --gpu-memory-utilization 0.88 \
  --max-model-len "${MAX_MODEL_LEN:-131072}" \
  "$@" >/dev/null

echo "container started; logs: docker logs -f $NAME"
