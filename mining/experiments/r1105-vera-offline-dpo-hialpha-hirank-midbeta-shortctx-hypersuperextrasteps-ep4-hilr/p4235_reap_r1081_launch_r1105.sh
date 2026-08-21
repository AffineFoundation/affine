#!/usr/bin/env bash
# p4235 host-side: R1081 REFUTE → reap r252 :8003 GPUs6,7 → R1105 Hyper HiLR TRAIN
# Do not touch R1090 train on GPUs 4,5 / teacher:8000 / king:8001. Never pkill -f.
set -euo pipefail
ROOT=/home/const/subnet120/mining/experiments
source /home/const/subnet120/.venv/bin/activate
cd /home/const/subnet120/mining
EXP=r1105-vera-offline-dpo-hialpha-hirank-midbeta-shortctx-hypersuperextrasteps-ep4-hilr
# (body already executed this pass; kept for replay / next-axis template)
echo "see pass 4235 status.log — already armed train.pid on r252"
