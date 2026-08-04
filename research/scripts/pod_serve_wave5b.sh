#!/bin/bash
# Wave-5b: remaining unscored kings with swe scores (VI already in 5a; do XLVIII L X XLIX XC + download VI swap).
# Call after wave-5a finishes; downloads extra models then serves.
set -euo pipefail
source /root/venv/bin/activate
cd /root
export HF_HOME=/root/hf
export PATH="/usr/local/cuda/bin:$PATH"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONPATH=/root
source /root/.env.hf || true

python - <<'PY'
from huggingface_hub import snapshot_download
import os
tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
for repo in [
    "dendriteholdings/albedo-qwen3.6-35b-king-VI",
    "dendriteholdings/albedo-qwen3.6-35b-king-XLVIII",
    "dendriteholdings/albedo-qwen3.6-35b-king-L",
    "dendriteholdings/albedo-qwen3.6-35b-king-X",
    "dendriteholdings/albedo-qwen3.6-35b-king-XLIX",
    "dendriteholdings/albedo-qwen3.6-35b-king-XC",
]:
    print("START", repo, flush=True)
    snapshot_download(repo, token=tok)
    print("DONE", repo, flush=True)
print("W5B_DL_DONE", flush=True)
PY

# Kill king servers (keep teacher if up)
pkill -f "albedo-qwen3.6-35b-king" || true
sleep 5
for port in 8001 8002 8003 8004 8005 8006; do fuser -k ${port}/tcp 2>/dev/null || true; done
sleep 2

if ! curl -sf -m 2 http://127.0.0.1:8000/v1/models >/dev/null; then
  CUDA_VISIBLE_DEVICES=0,1 nohup vllm serve zai-org/GLM-4.5-Air-FP8 \
    --port 8000 --tensor-parallel-size 2 --max-model-len 32768 \
    --gpu-memory-utilization 0.85 --max-num-batched-tokens 2048 \
    > /root/logs/serve_teacher.log 2>&1 &
fi

python -m harness.serve --kings \
  genesis:8001:2 VI:8002:3 XLVIII:8003:4 L:8004:5 X:8005:6 XLIX:8006:7

for i in $(seq 1 180); do
  curl -sf -m 2 http://127.0.0.1:8000/v1/models >/dev/null && break
  sleep 10
done
for port in 8001 8002 8003 8004 8005 8006; do
  for i in $(seq 1 120); do
    curl -sf -m 2 http://127.0.0.1:$port/v1/models >/dev/null && break
    sleep 10
  done
done

nohup python -u -m harness.runner \
  --turns /root/data/turns_minicoder.jsonl \
  --miners king-genesis:8001 king-VI:8002 king-XLVIII:8003 king-L:8004 king-X:8005 king-XLIX:8006 \
  --n-turns 200 \
  --out /root/results/ekings_w5b.jsonl \
  --ref-cache /root/results/ref_minicoder.jsonl \
  --concurrency 16 \
  > /root/logs/ekings_w5b.log 2>&1 &
echo RUNNER_W5B_PID=$!
