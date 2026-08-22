#!/bin/bash
# Phase 2 serving: 2x G (Qwen3-4B, runtime LoRA) + 2x frozen D (Qwen3-8B+LoRA).
# GPU 2 stays free for the loop's trainer, GPU 3 for checkpoint SWE serving,
# GPUs 6-7 spare (optional retrained-D diagnostic).
set -x
export HF_TOKEN=$(cat /root/.hf_token)
export HF_HOME=/opt/hf
export PATH=/opt/venv/bin:$PATH
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
DLORA=/root/work/trackC/discD/lora
mkdir -p /root/work/logs

CUDA_VISIBLE_DEVICES=0 nohup vllm serve Qwen/Qwen3-4B \
  --port 8002 --max-model-len 32768 --gpu-memory-utilization 0.92 \
  --enable-lora --max-lora-rank 32 --max-loras 4 \
  > /root/work/logs/g_srv0.log 2>&1 &

CUDA_VISIBLE_DEVICES=4 nohup vllm serve Qwen/Qwen3-4B \
  --port 8004 --max-model-len 32768 --gpu-memory-utilization 0.92 \
  --enable-lora --max-lora-rank 32 --max-loras 4 \
  > /root/work/logs/g_srv1.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup vllm serve Qwen/Qwen3-8B \
  --port 8003 --max-model-len 8192 --gpu-memory-utilization 0.92 \
  --enable-lora --max-lora-rank 16 \
  --lora-modules discD=$DLORA \
  > /root/work/logs/d_srv0.log 2>&1 &

CUDA_VISIBLE_DEVICES=5 nohup vllm serve Qwen/Qwen3-8B \
  --port 8005 --max-model-len 8192 --gpu-memory-utilization 0.92 \
  --enable-lora --max-lora-rank 16 \
  --lora-modules discD=$DLORA \
  > /root/work/logs/d_srv1.log 2>&1 &

echo "phase2 servers launching (G: 8002, 8004; D: 8003, 8005)"
