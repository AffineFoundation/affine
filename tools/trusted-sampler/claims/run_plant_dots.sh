#!/usr/bin/env bash
# usage: run_plant_dots.sh GPU EPOCH [SHARD NSHARDS]
cd /root/trusted-sampler; g=$1; e=$2; s=${3:-0}; n=${4:-1}; PY=/root/sampler-venv/bin/python
while [ "$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 2000 ]; do sleep 5; done
CUDA_VISIBLE_DEVICES=$g $PY claims/plant_dots.py --epoch $e --shard $s --nshards $n
