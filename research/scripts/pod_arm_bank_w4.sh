#!/bin/bash
# After ekings_w4 runner exits, rescore prior-bank for wave-4 miners.
set -euo pipefail
# shellcheck disable=SC1091
source /root/venv/bin/activate
cd /root
export HF_HOME=/root/hf
LOG=/root/logs/bank_w4.log
echo "watcher_start $(date)" >>"$LOG"
while pgrep -f "python -u -m harness.runner" >/dev/null; do
  echo "waiting_runner $(date)" >>"$LOG"
  sleep 60
done
echo "runner_done $(date)" >>"$LOG"
wc -l /root/results/ekings_w4.jsonl >>"$LOG"
exec python -u -m harness.rescore_bank \
  --src /root/results/ekings_w4.jsonl \
  --out /root/results/bank_w4.jsonl \
  --turns /root/data/turns_minicoder.jsonl \
  --ref-cache /root/results/ref_minicoder.jsonl \
  --n-turns 200 --concurrency 32 --turn-concurrency 6
