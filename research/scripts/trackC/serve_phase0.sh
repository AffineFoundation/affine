#!/bin/bash
# Phase 0/1 serving: two Qwen3-32B teacher replicas (TP=2) + G0 Qwen3-4B.
set -x
export HF_TOKEN=$(cat /root/.hf_token)
export HF_HOME=/opt/hf
export PATH=/opt/venv/bin:$PATH
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
mkdir -p /root/work/logs

CUDA_VISIBLE_DEVICES=0,1 nohup vllm serve Qwen/Qwen3-32B \
  --port 8010 --tensor-parallel-size 2 --max-model-len 32768 \
  --gpu-memory-utilization 0.92 \
  > /root/work/logs/teacher0.log 2>&1 &

CUDA_VISIBLE_DEVICES=2,3 nohup vllm serve Qwen/Qwen3-32B \
  --port 8011 --tensor-parallel-size 2 --max-model-len 32768 \
  --gpu-memory-utilization 0.92 \
  > /root/work/logs/teacher1.log 2>&1 &

CUDA_VISIBLE_DEVICES=4 nohup vllm serve Qwen/Qwen3-4B \
  --port 8002 --max-model-len 32768 \
  --gpu-memory-utilization 0.92 \
  > /root/work/logs/g0.log 2>&1 &

echo "phase0 servers launching (8010, 8011 teacher; 8002 G0)"
