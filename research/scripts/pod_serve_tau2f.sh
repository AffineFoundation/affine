#!/bin/bash
# Serve teacher + 8 kings for D_tau2 force-only probe (affv-e11).
set -euo pipefail
source /root/venv/bin/activate
export HF_HOME=/root/hf
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p /root/logs

# Teacher on 0,1
CUDA_VISIBLE_DEVICES=0,1 nohup vllm serve zai-org/GLM-4.5-Air-FP8 \
  --port 8000 --tensor-parallel-size 2 --max-model-len 32768 \
  --gpu-memory-utilization 0.85 --max-num-batched-tokens 2048 \
  > /root/logs/vllm_teacher.log 2>&1 &

# Kings one GPU each on 2..7 + reuse if needed — 6 kings on GPUs 2-7
CUDA_VISIBLE_DEVICES=2 nohup vllm serve dendriteholdings/albedo-qwen3.6-35b-king-genesis \
  --port 8001 --tensor-parallel-size 1 --max-model-len 16384 \
  --gpu-memory-utilization 0.90 --max-num-batched-tokens 2048 \
  > /root/logs/vllm_genesis.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup vllm serve dendriteholdings/albedo-qwen3.6-35b-king-II \
  --port 8002 --tensor-parallel-size 1 --max-model-len 16384 \
  --gpu-memory-utilization 0.90 --max-num-batched-tokens 2048 \
  > /root/logs/vllm_II.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup vllm serve dendriteholdings/albedo-qwen3.6-35b-king-XCIX \
  --port 8003 --tensor-parallel-size 1 --max-model-len 16384 \
  --gpu-memory-utilization 0.90 --max-num-batched-tokens 2048 \
  > /root/logs/vllm_XCIX.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup vllm serve dendriteholdings/albedo-qwen3.6-35b-king-VIII \
  --port 8004 --tensor-parallel-size 1 --max-model-len 16384 \
  --gpu-memory-utilization 0.90 --max-num-batched-tokens 2048 \
  > /root/logs/vllm_VIII.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup vllm serve dendriteholdings/albedo-qwen3.6-35b-king-XL \
  --port 8005 --tensor-parallel-size 1 --max-model-len 16384 \
  --gpu-memory-utilization 0.90 --max-num-batched-tokens 2048 \
  > /root/logs/vllm_XL.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup vllm serve dendriteholdings/albedo-qwen3.6-35b-king-XLVI \
  --port 8006 --tensor-parallel-size 1 --max-model-len 16384 \
  --gpu-memory-utilization 0.90 --max-num-batched-tokens 2048 \
  > /root/logs/vllm_XLVI.log 2>&1 &

echo "launched; waiting for /health"
for port in 8000 8001 8002 8003 8004 8005 8006; do
  for i in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1 \
       || curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      echo "READY $port"
      break
    fi
    sleep 10
  done
done
echo SERVE_READY
