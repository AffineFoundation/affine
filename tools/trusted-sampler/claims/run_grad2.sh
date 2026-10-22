#!/usr/bin/env bash
cd /root/trusted-sampler; g=$1
if [ "$g" != 7 ]; then CUDA_VISIBLE_DEVICES=$g /root/sampler-venv/bin/python claims/grad_claims.py --epoch 62 --shard $g --nshards 7; fi
while [ "$(ls claims/data/e64/gen-*.json 2>/dev/null | wc -l)" -lt 8 ]; do sleep 15; done
while [ "$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 2000 ]; do sleep 10; done
CUDA_VISIBLE_DEVICES=$g /root/sampler-venv/bin/python claims/grad_claims.py --epoch 64 --shard $g --nshards 8
