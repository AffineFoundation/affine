#!/bin/bash
# After tau2n_a reaches ≥500 lines (6×~100), launch wave B.
set -euo pipefail
TARGET=${1:-500}
echo "waiting for tau2n_a ≥$TARGET lines..."
while true; do
  n=0
  [[ -f /root/results/ekings_tau2n_a.jsonl ]] && n=$(wc -l </root/results/ekings_tau2n_a.jsonl)
  echo "tau2n_a lines=$n $(date -u +%H:%M:%S)"
  [[ "$n" -ge "$TARGET" ]] && break
  sleep 60
done
# ensure runner finished writing
while pgrep -f "ekings_tau2n_a.jsonl" >/dev/null; do sleep 15; done
echo "TAU2N_A_DONE lines=$(wc -l </root/results/ekings_tau2n_a.jsonl)"
bash /root/pod_serve_tau2n_b.sh
while true; do
  n=0
  [[ -f /root/results/ekings_tau2n_b.jsonl ]] && n=$(wc -l </root/results/ekings_tau2n_b.jsonl)
  echo "tau2n_b lines=$n $(date -u +%H:%M:%S)"
  [[ "$n" -ge 150 ]] && break
  sleep 60
done
while pgrep -f "ekings_tau2n_b.jsonl" >/dev/null; do sleep 15; done
echo "TAU2N_B_DONE lines=$(wc -l </root/results/ekings_tau2n_b.jsonl)"
echo ALL_TAU2N_DONE
