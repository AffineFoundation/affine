#!/usr/bin/env bash
cd /root/trusted-sampler
for g in 0 1 2 3 4 5 6 7; do
  setsid nohup bash -c "sleep $((g*40)); exec claims/run_gen.sh $g" > logs/claims-gen-g$g.log 2>&1 < /dev/null &
done
