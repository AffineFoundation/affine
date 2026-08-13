#!/usr/bin/env bash
# Launch two GLM-Air TP=4 replicas sequentially (A:40000 then B:40001).
# Run on the teacher pod. Separate /tmp caches avoid gocryptfs + compile races.
set -euo pipefail

REPO=${TEACHER_REPO:-zai-org/GLM-4.5-Air-FP8}
BATCHED=${MAX_NUM_BATCHED_TOKENS:-8192}
UTIL=${GPU_MEMORY_UTILIZATION:-0.85}
MAX_LEN=${MAX_MODEL_LEN:-65536}

launch_one() {
  local name=$1 port=$2 gpus=$3
  local cache=/tmp/vllm_${name}
  mkdir -p "$cache"/{triton,inductor,xdg,vllm} /root/logs
  # shellcheck disable=SC1091
  source /root/venv/bin/activate
  export HF_HOME=/root/hf HUGGINGFACE_HUB_CACHE=/root/hf/hub
  local cuda=/root/venv/lib/python3.12/site-packages/nvidia/cu13
  export CUDA_HOME=$cuda CUDA_PATH=$cuda PATH=$cuda/bin:$PATH
  export LD_LIBRARY_PATH=$cuda/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
  export LIBRARY_PATH=$cuda/lib${LIBRARY_PATH:+:$LIBRARY_PATH}
  export XDG_CACHE_HOME=$cache/xdg VLLM_CACHE_ROOT=$cache/vllm
  export TRITON_CACHE_DIR=$cache/triton TORCHINDUCTOR_CACHE_DIR=$cache/inductor
  export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ALLREDUCE_USE_FLASHINFER=0
  export VLLM_USE_FLASHINFER_MOE_FP16=0 VLLM_USE_FLASHINFER_MOE_FP8=0
  export VLLM_USE_FLASHINFER_MOE_FP4=0
  export CUDA_VISIBLE_DEVICES=$gpus
  : >"/root/logs/vllm_teacher_${name}.log"
  nohup vllm serve "$REPO" --host 0.0.0.0 --port "$port" \
    --tensor-parallel-size 4 --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$UTIL" --max-num-batched-tokens "$BATCHED" \
    --attention-backend FLASH_ATTN \
    --attention-config.use_trtllm_attention 0 \
    --compilation-config.pass_config.fuse_allreduce_rms false \
    --moe-backend triton \
    >"/root/logs/vllm_teacher_${name}.log" 2>&1 &
  echo $! >"/root/logs/vllm_teacher_${name}.pid"
  echo "launched $name pid=$(cat /root/logs/vllm_teacher_${name}.pid) :$port gpus=$gpus"
}

wait_port() {
  local port=$1 name=$2
  for i in $(seq 1 90); do
    if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null; then
      echo "READY $name :$port"
      return 0
    fi
    if ! kill -0 "$(cat /root/logs/vllm_teacher_${name}.pid)" 2>/dev/null; then
      echo "DEAD $name"; tail -n 40 "/root/logs/vllm_teacher_${name}.log"; return 1
    fi
    sleep 10
  done
  echo "TIMEOUT $name"; return 1
}

pkill -f 'vllm serve zai-org/GLM-4.5-Air-FP8' 2>/dev/null || true
sleep 4
launch_one a 40000 0,1,2,3
wait_port 40000 a
launch_one b 40001 4,5,6,7
wait_port 40001 b
echo DUAL_OK
