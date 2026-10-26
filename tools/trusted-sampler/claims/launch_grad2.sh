#!/usr/bin/env bash
cd /root/trusted-sampler
for g in 0 1 2 3 4 5 6 7; do setsid nohup claims/run_grad2.sh $g > logs/claims-grad-g$g.log 2>&1 < /dev/null & done
