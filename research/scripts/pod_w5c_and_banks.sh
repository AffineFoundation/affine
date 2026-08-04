#!/bin/bash
# Finish wave-5c (XC) + complete bank_w5a/w5b on teacher. XC on GPU 3.
set -euo pipefail
source /root/.env.hf || true
source /root/venv/bin/activate
export PYTHONPATH=/root
export HF_HOME=/root/hf
export PATH="/usr/local/cuda/bin:$PATH"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p /root/logs /root/results

pkill -f "albedo-qwen3.6-35b-king-XC" || true
fuser -k 8002/tcp 2>/dev/null || true
sleep 2

# XC on free GPU 3 (teacher 0,1; genesis 2)
nohup python -u -m harness.serve --kings XC:8002:3 \
  > /root/logs/serve_xc_launch.log 2>&1 &
echo SERVE_XC=$!

# Teacher-only bank jobs (resume-safe)
nohup python -u -m harness.rescore_bank \
  --src /root/results/ekings_w5a.jsonl \
  --out /root/results/bank_w5a.jsonl \
  --turns /root/data/turns_minicoder.jsonl \
  --ref-cache /root/results/ref_minicoder.jsonl \
  --n-turns 200 --concurrency 20 --turn-concurrency 6 \
  > /root/logs/bank_w5a.log 2>&1 &
echo BANK_W5A=$!

nohup python -u -m harness.rescore_bank \
  --src /root/results/ekings_w5b.jsonl \
  --out /root/results/bank_w5b.jsonl \
  --turns /root/data/turns_minicoder.jsonl \
  --ref-cache /root/results/ref_minicoder.jsonl \
  --n-turns 200 --concurrency 20 --turn-concurrency 6 \
  > /root/logs/bank_w5b.log 2>&1 &
echo BANK_W5B=$!

# Arm XC runner once ready
nohup bash -lc '
  for i in $(seq 1 120); do
    if curl -sf -m 2 http://127.0.0.1:8002/v1/models >/dev/null; then
      echo XC_OK at $i
      break
    fi
    sleep 10
  done
  if ! curl -sf -m 2 http://127.0.0.1:8002/v1/models >/dev/null; then
    echo XC_FAIL
    tail -40 /root/logs/serve_king-XC.log
    exit 1
  fi
  python -u -m harness.runner \
    --turns /root/data/turns_minicoder.jsonl \
    --miners king-XC:8002 \
    --n-turns 200 \
    --out /root/results/ekings_w5c.jsonl \
    --ref-cache /root/results/ref_minicoder.jsonl \
    --concurrency 12 \
    > /root/logs/ekings_w5c.log 2>&1
' > /root/logs/w5c_arm.log 2>&1 &
echo ARM=$!

sleep 5
echo "=== launch logs ==="
tail -3 /root/logs/serve_xc_launch.log /root/logs/bank_w5a.log /root/logs/bank_w5b.log /root/logs/w5c_arm.log 2>/dev/null || true
nvidia-smi --query-gpu=index,memory.used --format=csv
echo ARMED_OK
