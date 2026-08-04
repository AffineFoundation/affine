#!/bin/bash
# Wait for tau2a *results file* to reach n turns, then launch tau2b.
set -euo pipefail
TARGET=${1:-100}
echo "waiting for tau2a ≥$TARGET lines..."
while true; do
  n=0
  [[ -f /root/results/ekings_tau2a.jsonl ]] && n=$(wc -l </root/results/ekings_tau2a.jsonl)
  echo "tau2a lines=$n $(date -u +%H:%M:%S)"
  if [[ "$n" -ge "$TARGET" ]]; then
    break
  fi
  # still loading/running?
  if ! pgrep -f "harness.serve --kings" >/dev/null \
     && ! pgrep -f "ekings_tau2a.jsonl" >/dev/null \
     && [[ "$n" -eq 0 ]]; then
    echo "WARN: no tau2a activity yet; keep waiting"
  fi
  sleep 60
done
echo "TAU2A_DONE $(date -u +%H:%M:%S) lines=$(wc -l </root/results/ekings_tau2a.jsonl)"
bash /root/pod_serve_tau2b.sh
echo "TAU2B_ARMED"
while true; do
  n=0
  [[ -f /root/results/ekings_tau2b.jsonl ]] && n=$(wc -l </root/results/ekings_tau2b.jsonl)
  echo "tau2b lines=$n $(date -u +%H:%M:%S)"
  [[ "$n" -ge "$TARGET" ]] && break
  sleep 60
done
echo "TAU2B_DONE $(date -u +%H:%M:%S) lines=$(wc -l </root/results/ekings_tau2b.jsonl)"
echo ALL_TAU2_DONE
