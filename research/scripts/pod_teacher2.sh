#!/bin/bash
# Second-teacher replication: Qwen3-32B as teacher C2 on 8 kings × 100 turns.
# Run AFTER RT-3b live duels finish (frees GPUs). Kill GLM serves first.
set -euo pipefail
# shellcheck disable=SC1091
source /root/venv/bin/activate
cd /root
export HF_HOME=/root/hf
export PATH="/usr/local/cuda/bin:$PATH"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p /root/logs /root/results

# Stop prior vLLM
pkill -f "vllm serve" || true
sleep 5
for port in 8000 8001 8002 8003 8004 8005 8006 8007; do
  fuser -k ${port}/tcp 2>/dev/null || true
done
sleep 2

# Teacher2: Qwen3-32B on GPUs 0,1
CUDA_VISIBLE_DEVICES=0,1 nohup vllm serve Qwen/Qwen3-32B \
  --port 8000 --tensor-parallel-size 2 --max-model-len 32768 \
  --gpu-memory-utilization 0.90 --max-num-batched-tokens 2048 \
  > /root/logs/serve_teacher2.log 2>&1 &

python -m harness.serve --kings \
  genesis:8001:2 I:8002:3 II:8003:4 XCIX:8004:5 VII:8005:6 VIII:8006:7

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

# Override teacher endpoint via env if runner supports it; else patch config.
# harness.runner uses TEACHER from config — temporarily rewrite config on pod.
python - <<'PY'
from pathlib import Path
p = Path("/root/harness/config.py")
text = p.read_text()
text2 = text.replace(
    'TEACHER = ModelCfg(\n    name="glm45air",\n    repo="zai-org/GLM-4.5-Air-FP8",\n    port=8000,\n    family="glm",\n    gpus="0,1",\n    tp=2,\n)',
    'TEACHER = ModelCfg(\n    name="qwen3-32b",\n    repo="Qwen/Qwen3-32B",\n    port=8000,\n    family="qwen",\n    gpus="0,1",\n    tp=2,\n)',
)
if text2 == text:
    # fallback: replace repo string only
    text2 = text.replace("zai-org/GLM-4.5-Air-FP8", "Qwen/Qwen3-32B").replace(
        'name="glm45air"', 'name="qwen3-32b"').replace('family="glm"', 'family="qwen"')
p.write_text(text2)
print("TEACHER rewritten to Qwen3-32B")
PY

nohup python -u -m harness.runner \
  --turns /root/data/turns_minicoder.jsonl \
  --miners king-genesis:8001 king-I:8002 king-II:8003 king-XCIX:8004 king-VII:8005 king-VIII:8006 \
  --n-turns 100 \
  --out /root/results/ekings_teacher2_qwen32.jsonl \
  --ref-cache /root/results/ref_teacher2_qwen32.jsonl \
  --concurrency 12 \
  > /root/logs/ekings_teacher2.log 2>&1 &
echo TEACHER2_PID=$!
sleep 3
tail -5 /root/logs/ekings_teacher2.log
