#!/usr/bin/env bash
cd /root/trusted-sampler; g=$1
while [ "$(ls claims/data/e62/gen-*.json claims/data/e64/gen-*.json 2>/dev/null | wc -l)" -lt 16 ]; do sleep 15; done
for e in 62 64; do
  while [ "$(ls claims/data/e$e/gen-*.json 2>/dev/null | wc -l)" -lt 8 ]; do sleep 15; done
  while [ "$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 2000 ]; do sleep 10; done
  CUDA_VISIBLE_DEVICES=$g /root/sampler-venv/bin/python claims/grad_claims.py --epoch $e --shard $g
done
