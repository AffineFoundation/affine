#!/bin/bash
# After ekings_w5c finishes: complete bank_w5a then bank_w5b (teacher-only).
set -euo pipefail
source /root/.env.hf || true
source /root/venv/bin/activate
export PYTHONPATH=/root

echo "waiting for w5c runner..."
while pgrep -f "harness.runner .*ekings_w5c" >/dev/null; do
  sleep 30
done
echo "W5C_DONE $(date -u +%H:%M:%S) lines=$(wc -l </root/results/ekings_w5c.jsonl)"

# Optional: free genesis GPU — not needed for bank
# kill genesis serve if desired

python -u -m harness.rescore_bank \
  --src /root/results/ekings_w5a.jsonl \
  --out /root/results/bank_w5a.jsonl \
  --turns /root/data/turns_minicoder.jsonl \
  --ref-cache /root/results/ref_minicoder.jsonl \
  --n-turns 200 --concurrency 48 --turn-concurrency 12 \
  > /root/logs/bank_w5a.log 2>&1
echo "W5A_BANK_DONE $(date -u +%H:%M:%S) lines=$(wc -l </root/results/bank_w5a.jsonl)"

python -u -m harness.rescore_bank \
  --src /root/results/ekings_w5b.jsonl \
  --out /root/results/bank_w5b.jsonl \
  --turns /root/data/turns_minicoder.jsonl \
  --ref-cache /root/results/ref_minicoder.jsonl \
  --n-turns 200 --concurrency 48 --turn-concurrency 12 \
  > /root/logs/bank_w5b.log 2>&1
echo "W5B_BANK_DONE $(date -u +%H:%M:%S) lines=$(wc -l </root/results/bank_w5b.jsonl)"

# Also bank XC if present
if [[ -s /root/results/ekings_w5c.jsonl ]]; then
  python -u -m harness.rescore_bank \
    --src /root/results/ekings_w5c.jsonl \
    --out /root/results/bank_w5c.jsonl \
    --turns /root/data/turns_minicoder.jsonl \
    --ref-cache /root/results/ref_minicoder.jsonl \
    --n-turns 200 --concurrency 48 --turn-concurrency 12 \
    > /root/logs/bank_w5c.log 2>&1
  echo "W5C_BANK_DONE $(date -u +%H:%M:%S) lines=$(wc -l </root/results/bank_w5c.jsonl)"
fi

echo ALL_BANKS_DONE
