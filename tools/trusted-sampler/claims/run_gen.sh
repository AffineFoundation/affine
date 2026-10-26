#!/usr/bin/env bash
# run gen for both epochs on one GPU
cd /root/trusted-sampler; g=$1
while [ "$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 2000 ]; do sleep 10; done
for e in 62 64; do CUDA_VISIBLE_DEVICES=$g /root/sampler-venv/bin/python claims/gen_claims.py --epoch $e --shard $g; done
