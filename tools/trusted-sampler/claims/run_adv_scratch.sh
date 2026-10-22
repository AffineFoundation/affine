#!/bin/bash
# GPU EPOCH SHARD NSHARDS MODES [IDS]
cd /root/trusted-sampler; source /root/sampler-venv/bin/activate; export CUDA_VISIBLE_DEVICES=$1
python claims/adv_scratch.py --epoch $2 --shard $3 --nshards $4 --modes $5 ${6:+--ids $6}
