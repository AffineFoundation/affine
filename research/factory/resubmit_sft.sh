#!/bin/bash
# Wait for each SFT step250 checkpoint on the bench box to be complete
# (tokenizer present + size stable across two checks), then bench it.
set -u
BOX="ssh -n -o StrictHostKeyChecking=no -o ConnectTimeout=15 -p 20300 root@REDACTED_IP"
cd /home/const/subnet120/research/factory

declare -A done
for i in $(seq 1 60); do
  for arm in witness ctrl worldmodel; do
    [ "${done[$arm]:-}" = "1" ] && continue
    d=/root/incoming/${arm}_step250
    s1=$($BOX "test -f $d/tokenizer.json && du -sb $d | cut -f1" 2>/dev/null)
    [ -z "$s1" ] && { echo "$arm: not ready (no tokenizer yet)"; continue; }
    sleep 45
    s2=$($BOX "du -sb $d | cut -f1" 2>/dev/null)
    if [ "$s1" = "$s2" ]; then
      echo "$arm: stable at $s2 bytes, submitting bench"
      python3 factory_bench.py --model-path "$d" --arm "$arm" --step 250 \
        > /tmp/bench_${arm}250_retry.log 2>&1
      echo "$arm: bench done: $(tail -1 /tmp/bench_${arm}250_retry.log)"
      done[$arm]=1
    else
      echo "$arm: still growing ($s1 -> $s2)"
    fi
  done
  [ "${done[witness]:-}" = "1" ] && [ "${done[ctrl]:-}" = "1" ] && \
    [ "${done[worldmodel]:-}" = "1" ] && { echo ALL_SFT_BENCHED; exit 0; }
  sleep 90
done
echo RESUBMIT_TIMEOUT
exit 1
