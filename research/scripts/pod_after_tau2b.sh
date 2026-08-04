#!/bin/bash
# After tau2b finishes: run wave A (6 kings × 100 turns on D_tau2).
set -euo pipefail
echo "waiting for tau2b runner..."
while pgrep -f "ekings_tau2b.jsonl" >/dev/null; do
  sleep 30
done
echo "TAU2B_DONE lines=$(wc -l </root/results/ekings_tau2b.jsonl)"
bash /root/pod_serve_tau2a.sh
echo TAU2A_ARMED
while true; do
  n=0
  [[ -f /root/results/ekings_tau2a.jsonl ]] && n=$(wc -l </root/results/ekings_tau2a.jsonl)
  echo "tau2a lines=$n $(date -u +%H:%M:%S)"
  # 6 miners × 100 turns = 600 lines
  [[ "$n" -ge 600 ]] && break
  sleep 60
done
echo "TAU2A_DONE lines=$(wc -l </root/results/ekings_tau2a.jsonl)"
echo ALL_TAU2_DONE
