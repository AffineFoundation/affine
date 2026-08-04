#!/bin/bash
# Wait for bank_w5a to finish, then run bank_w5b (teacher-only, resume-safe).
set -euo pipefail
source /root/.env.hf || true
source /root/venv/bin/activate
export PYTHONPATH=/root

echo "waiting for bank_w5a..."
while pgrep -f "rescore_bank --src /root/results/ekings_w5a" >/dev/null; do
  sleep 30
done
echo "W5A_DONE $(date -u +%H:%M:%S) lines=$(wc -l </root/results/bank_w5a.jsonl)"

python -u -m harness.rescore_bank \
  --src /root/results/ekings_w5b.jsonl \
  --out /root/results/bank_w5b.jsonl \
  --turns /root/data/turns_minicoder.jsonl \
  --ref-cache /root/results/ref_minicoder.jsonl \
  --n-turns 200 --concurrency 48 --turn-concurrency 12 \
  > /root/logs/bank_w5b.log 2>&1

echo "W5B_DONE $(date -u +%H:%M:%S) lines=$(wc -l </root/results/bank_w5b.jsonl)"
