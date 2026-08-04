#!/bin/bash
set -euo pipefail
source /root/venv/bin/activate
cd /root
export HF_HOME=/root/hf
# stop prior rescore only
for pid in $(pgrep -f 'python -u -m harness.rescore_bank' || true); do
  kill "$pid" 2>/dev/null || true
done
sleep 2
nohup python -u -m harness.rescore_bank \
  --src /root/results/ekings_w2_v2_fullz.jsonl \
  --out /root/results/bank_w2_fullz.jsonl \
  --turns /root/data/turns_minicoder.jsonl \
  --ref-cache /root/results/ref_minicoder.jsonl \
  --n-turns 200 --concurrency 32 --turn-concurrency 6 \
  >> /root/logs/bank_w2_fullz.log 2>&1 &
echo RESTART_PID=$!
sleep 3
wc -l /root/results/bank_w2_fullz.jsonl
tail -5 /root/logs/bank_w2_fullz.log
