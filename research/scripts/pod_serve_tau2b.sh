#!/bin/bash
# D_tau2 wave B: remaining specialists (keep teacher + genesis for ref continuity).
set -euo pipefail
source /root/.env.hf || true
source /root/venv/bin/activate
export PYTHONPATH=/root HF_HOME=/root/hf
export PATH="/usr/local/cuda/bin:$PATH"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_FLASHINFER_SAMPLER=0

# Kill king servers only (keep teacher on :8000 if healthy)
pkill -f "albedo-qwen3.6-35b-king" || true
sleep 5
for port in 8001 8002 8003 8004 8005 8006; do
  fuser -k ${port}/tcp 2>/dev/null || true
done
sleep 2

if ! curl -sf -m 2 http://127.0.0.1:8000/v1/models >/dev/null; then
  CUDA_VISIBLE_DEVICES=0,1 nohup vllm serve zai-org/GLM-4.5-Air-FP8 \
    --port 8000 --tensor-parallel-size 2 --max-model-len 32768 \
    --gpu-memory-utilization 0.85 --max-num-batched-tokens 2048 \
    > /root/logs/serve_teacher.log 2>&1 &
fi

python -m harness.serve --kings \
  genesis:8001:2 XLVI:8002:3 CI:8003:4

for port in 8001 8002 8003; do
  for i in $(seq 1 120); do
    curl -sf -m 2 http://127.0.0.1:$port/v1/models >/dev/null && break
    sleep 10
  done
done

nohup python -u -m harness.runner \
  --turns /root/data/turns_tau2.jsonl \
  --miners king-XLVI:8002 king-CI:8003 \
  --n-turns 100 \
  --out /root/results/ekings_tau2b.jsonl \
  --ref-cache /root/results/ref_tau2.jsonl \
  --concurrency 12 \
  > /root/logs/ekings_tau2b.log 2>&1 &
echo RUNNER_TAU2B=$!
