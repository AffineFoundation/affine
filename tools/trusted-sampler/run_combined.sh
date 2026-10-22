#!/usr/bin/env bash
# usage: run_combined.sh GPU PAIR ID   -- waits until the GPU is empty, then runs combined.py in its own session
cd /root/trusted-sampler; g=$1; pair=$2; id=$3
while [ "$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 2000 ]; do sleep 15; done
CUDA_VISIBLE_DEVICES=$g exec /root/sampler-venv/bin/python combined.py --pair $pair -K 128 -m 8 --id $id
