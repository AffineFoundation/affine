#!/usr/bin/env bash
# usage: run_gnorm.sh GPU EPOCH SHARD NSHARDS
cd /root/trusted-sampler; PY=/root/sampler-venv/bin/python
while [ "$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 2000 ]; do sleep 5; done
CUDA_VISIBLE_DEVICES=$1 $PY claims/gnorm.py --epoch $2 --shard $3 --nshards $4
