#!/bin/bash
# Serve teacher + genesis + XLII/XL/C/XCVI and run E-KINGS wave-4 (200 turns).
set -euo pipefail
# shellcheck disable=SC1091
source /root/venv/bin/activate
cd /root
export HF_HOME=/root/hf
export PATH="/usr/local/cuda/bin:$PATH"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p /root/logs /root/results

if ! curl -sf -m 2 http://127.0.0.1:8000/v1/models >/dev/null; then
  CUDA_VISIBLE_DEVICES=0,1 nohup vllm serve zai-org/GLM-4.5-Air-FP8 \
    --port 8000 --tensor-parallel-size 2 --max-model-len 32768 \
    --gpu-memory-utilization 0.85 --max-num-batched-tokens 2048 \
    > /root/logs/serve_teacher.log 2>&1 &
fi

python -m harness.serve --kings \
  genesis:8001:2 XLII:8002:3 XL:8003:4 C:8004:5 XCVI:8005:6

# Wait teacher
for i in $(seq 1 180); do
  curl -sf -m 2 http://127.0.0.1:8000/v1/models >/dev/null && break
  sleep 10
done

nohup python -u -m harness.runner \
  --turns /root/data/turns_minicoder.jsonl \
  --miners king-genesis:8001 king-XLII:8002 king-XL:8003 king-C:8004 king-XCVI:8005 \
  --n-turns 200 \
  --out /root/results/ekings_w4.jsonl \
  --ref-cache /root/results/ref_minicoder.jsonl \
  --concurrency 16 \
  > /root/logs/ekings_w4.log 2>&1 &
echo RUNNER_PID=$!
sleep 3
tail -5 /root/logs/ekings_w4.log
