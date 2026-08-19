#!/usr/bin/env bash
# R888: TP1 teacher on GPU0 (7-GPU pod). Train uses CUDA_VISIBLE_DEVICES=2,3.
set -euo pipefail
export PATH="/root/venv/bin:$HOME/.local/bin:$PATH"
# shellcheck disable=SC1091
source /root/venv/bin/activate
if [[ -f /root/mine.env ]]; then set -a; # shellcheck disable=SC1091
  source /root/mine.env; set +a; fi
export HF_HOME=${HF_HOME:-/root/hf}
unset HF_TOKEN || true
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
# Match other mine pods: cu13 libs from venv for nvcc/cuda_home issues
CU_LIB=$(python -c 'import nvidia.cu13, pathlib; print(pathlib.Path(nvidia.cu13.__file__).parent / "lib")' 2>/dev/null || true)
if [[ -n "${CU_LIB}" && -d "${CU_LIB}" ]]; then
  export CUDA_HOME="${CU_LIB%/lib}"
  export CUDA_PATH="$CUDA_HOME"
  export LD_LIBRARY_PATH="${CU_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

TEACHER_REV=f9a9c5acf5e543cd24d659a056c5dbcda78ffcfc
TEACHER_LOCAL=${TEACHER_LOCAL:-/root/hf/hub/models--zai-org--GLM-4.5-Air-FP8/snapshots/$TEACHER_REV}
test -e "$TEACHER_LOCAL/config.json"
mkdir -p /root/logs /root/affine_data

if curl -sf --max-time 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
  echo "[r888-teacher] already READY :8000"
  exit 0
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
# DeepGEMM / flashinfer MoE crash on this 7-GPU B200 image (p3981) — force triton path.
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
export VLLM_ALLREDUCE_USE_FLASHINFER=${VLLM_ALLREDUCE_USE_FLASHINFER:-0}
export VLLM_USE_FLASHINFER_MOE_FP16=${VLLM_USE_FLASHINFER_MOE_FP16:-0}
export VLLM_USE_FLASHINFER_MOE_FP8=${VLLM_USE_FLASHINFER_MOE_FP8:-0}
export VLLM_USE_FLASHINFER_MOE_FP4=${VLLM_USE_FLASHINFER_MOE_FP4:-0}
export VLLM_USE_DEEP_GEMM=${VLLM_USE_DEEP_GEMM:-0}
export VLLM_MOE_USE_DEEP_GEMM=${VLLM_MOE_USE_DEEP_GEMM:-0}
LOG=/root/logs/r888_teacher.nohup
: >"$LOG"
nohup env \
  VLLM_USE_FLASHINFER_SAMPLER="$VLLM_USE_FLASHINFER_SAMPLER" \
  VLLM_ALLREDUCE_USE_FLASHINFER="$VLLM_ALLREDUCE_USE_FLASHINFER" \
  VLLM_USE_FLASHINFER_MOE_FP16="$VLLM_USE_FLASHINFER_MOE_FP16" \
  VLLM_USE_FLASHINFER_MOE_FP8="$VLLM_USE_FLASHINFER_MOE_FP8" \
  VLLM_USE_FLASHINFER_MOE_FP4="$VLLM_USE_FLASHINFER_MOE_FP4" \
  VLLM_USE_DEEP_GEMM="$VLLM_USE_DEEP_GEMM" \
  VLLM_MOE_USE_DEEP_GEMM="$VLLM_MOE_USE_DEEP_GEMM" \
  /root/venv/bin/vllm serve "$TEACHER_LOCAL" \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 8192 \
  --attention-backend FLASH_ATTN \
  --attention-config.use_trtllm_attention 0 \
  --compilation-config.pass_config.fuse_allreduce_rms false \
  --moe-backend triton \
  --additional-config '{"gdn_prefill_backend": "triton"}' \
  --served-model-name zai-org/GLM-4.5-Air-FP8 \
  --enforce-eager \
  >>"$LOG" 2>&1 &
echo $! | tee /root/logs/r888_teacher.pid
echo "[r888-teacher] launched pid=$(cat /root/logs/r888_teacher.pid) CVD=$CUDA_VISIBLE_DEVICES TP=1 deep_gemm=0"
