#!/usr/bin/env bash
# Serve a policy checkpoint with vLLM on the B200 pod.
#
# This pod has the NVIDIA driver but no system CUDA toolkit, so flashinfer's
# JIT path cannot find nvcc on its own. nvcc ships inside the cu13 wheel that
# vLLM already pulls in, so we point CUDA_HOME at it and add ninja, which the
# JIT build also needs. Without both, engine start fails during kernel build.
#
# usage: pod_serve.sh <hf_repo> <port> <gpus> [extra vllm args...]
set -euo pipefail

REPO="${1:?hf repo}"
PORT="${2:-8002}"
GPUS="${3:-1,2}"
shift 3 || true

VENV=/root/vllmenv
CUDA_HOME="$VENV/lib/python3.12/site-packages/nvidia/cu13"
export CUDA_HOME
export CUDA_PATH="$CUDA_HOME"
# $VENV/bin must be on PATH too: the JIT build shells out to `ninja`, which
# lives there, and invoking vllm by absolute path does not add it.
export PATH="$CUDA_HOME/bin:$VENV/bin:$PATH"
export HF_HOME=/root/hf
export HF_TOKEN="$(cat /root/.hf_token 2>/dev/null || true)"

# flashinfer's attention kernels build fine here, but its sampling kernel does
# not, and it is the last thing touched during startup profiling. vLLM has a
# native torch sampler, so decline the flashinfer one rather than fight its JIT.
export VLLM_USE_FLASHINFER_SAMPLER=0

NGPU="$(echo "$GPUS" | tr ',' '\n' | grep -c .)"
LOG="/root/work/vllm_$(echo "$REPO" | tr '/' '_')_$PORT.log"

echo "serving $REPO on port $PORT, gpus=$GPUS (tp=$NGPU)"
echo "log: $LOG"

CUDA_VISIBLE_DEVICES="$GPUS" nohup "$VENV/bin/vllm" serve "$REPO" \
  --port "$PORT" \
  --tensor-parallel-size "$NGPU" \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.88 \
  --served-model-name "$REPO" \
  "$@" > "$LOG" 2>&1 &

echo "pid=$!"
