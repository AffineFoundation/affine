#!/bin/bash
# After ekings_w5a finishes: bank_w5a, then wave-5b, then bank_w5b.
set -euo pipefail
source /root/venv/bin/activate
source /root/.env.hf || true
export PYTHONPATH=/root
export HF_HOME=/root/hf
export PATH="/usr/local/cuda/bin:$PATH"

# Wait for w5a runner to finish
while pgrep -f "harness.runner.*ekings_w5a" >/dev/null; do
  n=$(wc -l < /root/results/ekings_w5a.jsonl 2>/dev/null || echo 0)
  echo "waiting w5a lines=$n"
  sleep 60
done
echo "W5A_DONE lines=$(wc -l < /root/results/ekings_w5a.jsonl)"

# Bank rescore for wave-5a miners (teacher still up)
python -u -m harness.rescore_bank \
  --src /root/results/ekings_w5a.jsonl \
  --out /root/results/bank_w5a.jsonl \
  --n-turns 200 \
  > /root/logs/bank_w5a.log 2>&1 || echo "bank_w5a failed (nonfatal)"

bash /root/pod_serve_wave5b.sh

# Wait w5b
while pgrep -f "harness.runner.*ekings_w5b" >/dev/null; do
  n=$(wc -l < /root/results/ekings_w5b.jsonl 2>/dev/null || echo 0)
  echo "waiting w5b lines=$n"
  sleep 60
done
echo "W5B_DONE lines=$(wc -l < /root/results/ekings_w5b.jsonl)"

python -u -m harness.rescore_bank \
  --src /root/results/ekings_w5b.jsonl \
  --out /root/results/bank_w5b.jsonl \
  --n-turns 200 \
  > /root/logs/bank_w5b.log 2>&1 || true

# Last king XC if downloaded
if [ -d /root/hf/hub ] || true; then
  python - <<'PY' || true
from huggingface_hub import snapshot_download
import os
tok=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
print("START XC", flush=True)
snapshot_download("dendriteholdings/albedo-qwen3.6-35b-king-XC", token=tok)
print("DONE XC", flush=True)
PY
fi

# Score XC alone if teacher+genesis still up
pkill -f "albedo-qwen3.6-35b-king-(VI|XLVIII|L|X|XLIX)" || true
sleep 3
for port in 8002 8003 8004 8005 8006; do fuser -k ${port}/tcp 2>/dev/null || true; done
sleep 2
python -m harness.serve --kings XC:8002:2
for i in $(seq 1 120); do
  curl -sf -m 2 http://127.0.0.1:8002/v1/models >/dev/null && break
  sleep 10
done
python -u -m harness.runner \
  --turns /root/data/turns_minicoder.jsonl \
  --miners king-XC:8002 \
  --n-turns 200 \
  --out /root/results/ekings_w5c.jsonl \
  --ref-cache /root/results/ref_minicoder.jsonl \
  --concurrency 12 \
  > /root/logs/ekings_w5c.log 2>&1 || true

python -u -m harness.rescore_bank \
  --src /root/results/ekings_w5c.jsonl \
  --out /root/results/bank_w5c.jsonl \
  --n-turns 200 \
  > /root/logs/bank_w5c.log 2>&1 || true

echo E6_ALL_WAVES_DONE
