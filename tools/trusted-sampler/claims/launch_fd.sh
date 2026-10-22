#!/usr/bin/env bash
cd /root/trusted-sampler
run() { g=$1; e=$2; u=$3; setsid nohup bash -c "sleep $((g*15)); CUDA_VISIBLE_DEVICES=$g exec /root/sampler-venv/bin/python claims/fd_check.py --epoch $e --uids $u" > logs/claims-fd-g$g.log 2>&1 < /dev/null & }
run 0 62 3; run 1 62 338; run 2 62 375; run 3 62 130
run 4 64 112; run 5 64 90; run 6 64 349; run 7 64 261
