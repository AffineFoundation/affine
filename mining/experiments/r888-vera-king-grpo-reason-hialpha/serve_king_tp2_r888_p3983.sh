#!/usr/bin/env bash
# p3983: TP1 king (reign36 vera) on R888 GPU 4 :8001. Do not touch teacher:8000 or GRPO 2,3.
# TP1 matches this 7-GPU pod's teacher path (avoids NCCL on non-contiguous free GPUs).
set -euo pipefail
export PATH="/root/venv/bin:$HOME/.local/bin:$PATH"
# shellcheck disable=SC1091
source /root/venv/bin/activate
if [[ -f /root/mine.env ]]; then set -a; # shellcheck disable=SC1091
  source /root/mine.env; set +a; fi
export HF_HOME=${HF_HOME:-/root/hf}
unset HF_TOKEN || true
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1

CU_LIB=$(python -c 'import nvidia.cu13, pathlib; print(pathlib.Path(nvidia.cu13.__file__).parent / "lib")' 2>/dev/null || true)
if [[ -n "${CU_LIB}" && -d "${CU_LIB}" ]]; then
  export CUDA_HOME="${CU_LIB%/lib}"
  export CUDA_PATH="$CUDA_HOME"
  export LD_LIBRARY_PATH="${CU_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

KING_REV=8e3f1695e058837ed80fec3238ff439fdc2d0f0e
KING_LOCAL=${KING_LOCAL:-/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/$KING_REV}
KING_NAME=vera6/affine-5g4yy75zuz-t6
test -e "$KING_LOCAL/config.json"
mkdir -p /root/logs /root/affine_data /root/.triton/cache/king_r888

if curl -sf --max-time 3 http://127.0.0.1:8001/v1/models >/dev/null 2>&1; then
  kid=$(curl -sf --max-time 3 http://127.0.0.1:8001/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
  echo "[r888-king] already READY :8001 id=$kid"
  exit 0
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4}
export TRITON_CACHE_DIR=/root/.triton/cache/king_r888
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
export VLLM_ALLREDUCE_USE_FLASHINFER=${VLLM_ALLREDUCE_USE_FLASHINFER:-0}
export VLLM_USE_FLASHINFER_MOE_FP16=${VLLM_USE_FLASHINFER_MOE_FP16:-0}
export VLLM_USE_FLASHINFER_MOE_FP8=${VLLM_USE_FLASHINFER_MOE_FP8:-0}
export VLLM_USE_FLASHINFER_MOE_FP4=${VLLM_USE_FLASHINFER_MOE_FP4:-0}
export VLLM_USE_DEEP_GEMM=${VLLM_USE_DEEP_GEMM:-0}
export VLLM_MOE_USE_DEEP_GEMM=${VLLM_MOE_USE_DEEP_GEMM:-0}

if [[ -z "$(ls -A /root/.triton/cache/king_r888 2>/dev/null || true)" ]]; then
  for cand in /root/.triton/cache/teacher /root/.triton/cache/chall_r888; do
    if [[ -d "$cand" ]] && [[ -n "$(ls -A "$cand" 2>/dev/null || true)" ]]; then
      cp -a "$cand"/. /root/.triton/cache/king_r888/ || true
      echo "[r888-king] seeded triton from $cand"
      break
    fi
  done
fi

LOG=/root/logs/r888_king_p3983.nohup
: >"$LOG"
nohup env \
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  VLLM_USE_FLASHINFER_SAMPLER="$VLLM_USE_FLASHINFER_SAMPLER" \
  VLLM_ALLREDUCE_USE_FLASHINFER="$VLLM_ALLREDUCE_USE_FLASHINFER" \
  VLLM_USE_FLASHINFER_MOE_FP16="$VLLM_USE_FLASHINFER_MOE_FP16" \
  VLLM_USE_FLASHINFER_MOE_FP8="$VLLM_USE_FLASHINFER_MOE_FP8" \
  VLLM_USE_FLASHINFER_MOE_FP4="$VLLM_USE_FLASHINFER_MOE_FP4" \
  VLLM_USE_DEEP_GEMM="$VLLM_USE_DEEP_GEMM" \
  VLLM_MOE_USE_DEEP_GEMM="$VLLM_MOE_USE_DEEP_GEMM" \
  /root/venv/bin/vllm serve "$KING_LOCAL" \
  --port 8001 \
  --tensor-parallel-size 1 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 8192 \
  --attention-backend FLASH_ATTN \
  --attention-config.use_trtllm_attention 0 \
  --compilation-config.pass_config.fuse_allreduce_rms false \
  --moe-backend triton \
  --additional-config '{"gdn_prefill_backend": "triton"}' \
  --served-model-name "$KING_NAME" \
  --enforce-eager \
  >>"$LOG" 2>&1 &
echo $! | tee /root/logs/r888_king_p3983.pid
echo "[r888-king] launched pid=$(cat /root/logs/r888_king_p3983.pid) CVD=$CUDA_VISIBLE_DEVICES TP=1 deep_gemm=0"
