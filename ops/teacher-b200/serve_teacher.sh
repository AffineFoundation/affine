#!/usr/bin/env bash
set -euo pipefail
source /root/venv/bin/activate
export HF_HOME=/root/hf HUGGINGFACE_HUB_CACHE=/root/hf/hub
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
CUDA_HOME_PIP=/root/venv/lib/python3.12/site-packages/nvidia/cu13
export CUDA_HOME="$CUDA_HOME_PIP" CUDA_PATH="$CUDA_HOME_PIP"
export PATH="$CUDA_HOME_PIP/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME_PIP/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBRARY_PATH="$CUDA_HOME_PIP/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export XDG_CACHE_HOME=/tmp/xdg_cache
export VLLM_CACHE_ROOT=/tmp/vllm_cache
export TRITON_CACHE_DIR=/tmp/triton_cache
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ALLREDUCE_USE_FLASHINFER=0
export VLLM_USE_FLASHINFER_MOE_FP16=0 VLLM_USE_FLASHINFER_MOE_FP8=0 VLLM_USE_FLASHINFER_MOE_FP4=0
mkdir -p "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" /root/logs
REPO=${TEACHER_REPO:-zai-org/GLM-4.5-Air-FP8}
PORT=${TEACHER_PORT:-40000}
TP=${TEACHER_TP:-8}
MAX_LEN=${MAX_MODEL_LEN:-65536}
GPU_UTIL=${GPU_MEMORY_UTILIZATION:-0.85}
BATCHED=${MAX_NUM_BATCHED_TOKENS:-8192}
LOG=/root/logs/vllm_teacher.log
echo "[serve] $(date -u +%Y-%m-%dT%H:%M:%SZ) repo=$REPO tp=$TP port=$PORT batched=$BATCHED util=$GPU_UTIL" | tee -a "$LOG"
exec vllm serve "$REPO" --host 0.0.0.0 --port "$PORT" --tensor-parallel-size "$TP" \
  --max-model-len "$MAX_LEN" --gpu-memory-utilization "$GPU_UTIL" \
  --max-num-batched-tokens "$BATCHED" --attention-backend FLASH_ATTN \
  --attention-config.use_trtllm_attention 0 \
  --compilation-config.pass_config.fuse_allreduce_rms false --moe-backend triton \
  >>"$LOG" 2>&1
