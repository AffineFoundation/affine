#!/bin/bash
# Arm bank_w2_fullz to start once harness.runner exits.
set -euo pipefail
source /root/venv/bin/activate
cd /root
export HF_HOME=/root/hf
LOG=/root/logs/bank_w2_fullz.log
mkdir -p /root/logs /root/results
echo "watcher_start $(date)" >>"$LOG"
while pgrep -f "python -u -m harness.runner" >/dev/null; do
  echo "waiting_runner $(date)" >>"$LOG"
  sleep 60
done
echo "runner_done $(date)" >>"$LOG"
wc -l /root/results/ekings_w2_v2_fullz.jsonl >>"$LOG"
exec python -u -m harness.rescore_bank \
  --src /root/results/ekings_w2_v2_fullz.jsonl \
  --out /root/results/bank_w2_fullz.jsonl \
  --turns /root/data/turns_minicoder.jsonl \
  --ref-cache /root/results/ref_minicoder.jsonl \
  --n-turns 200 --concurrency 48 --turn-concurrency 8
