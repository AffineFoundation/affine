#!/bin/bash
# Cap banks at 80 turns (enough for γ_bank). Stop full-200 w5a when ≥480 lines,
# then run w5b/w5c at --n-turns 80.
set -euo pipefail
source /root/.env.hf || true
source /root/venv/bin/activate
export PYTHONPATH=/root

TARGET=480
while true; do
  n=$(wc -l </root/results/bank_w5a.jsonl)
  echo "bank_w5a lines=$n $(date -u +%H:%M:%S)"
  if [[ "$n" -ge "$TARGET" ]]; then
    break
  fi
  # still running?
  if ! pgrep -f "rescore_bank --src /root/results/ekings_w5a" >/dev/null; then
    echo "w5a bank process ended early at $n lines"
    break
  fi
  sleep 60
done

# stop full-200 w5a if still going
pkill -f "rescore_bank --src /root/results/ekings_w5a" || true
sleep 3

# top up w5a to exactly first 80 turns (resume-safe)
python -u -m harness.rescore_bank \
  --src /root/results/ekings_w5a.jsonl \
  --out /root/results/bank_w5a.jsonl \
  --turns /root/data/turns_minicoder.jsonl \
  --ref-cache /root/results/ref_minicoder.jsonl \
  --n-turns 80 --concurrency 48 --turn-concurrency 12 \
  > /root/logs/bank_w5a_cap80.log 2>&1
echo "W5A_CAP80_DONE lines=$(wc -l </root/results/bank_w5a.jsonl)"

python -u -m harness.rescore_bank \
  --src /root/results/ekings_w5b.jsonl \
  --out /root/results/bank_w5b.jsonl \
  --turns /root/data/turns_minicoder.jsonl \
  --ref-cache /root/results/ref_minicoder.jsonl \
  --n-turns 80 --concurrency 48 --turn-concurrency 12 \
  > /root/logs/bank_w5b.log 2>&1
echo "W5B_CAP80_DONE lines=$(wc -l </root/results/bank_w5b.jsonl)"

python -u -m harness.rescore_bank \
  --src /root/results/ekings_w5c.jsonl \
  --out /root/results/bank_w5c.jsonl \
  --turns /root/data/turns_minicoder.jsonl \
  --ref-cache /root/results/ref_minicoder.jsonl \
  --n-turns 80 --concurrency 48 --turn-concurrency 12 \
  > /root/logs/bank_w5c.log 2>&1
echo "W5C_CAP80_DONE lines=$(wc -l </root/results/bank_w5c.jsonl)"
echo ALL_CAP80_DONE
