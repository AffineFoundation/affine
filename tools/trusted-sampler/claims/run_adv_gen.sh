#!/bin/bash
# GPU EPOCH : copies then distill (teacher = 066dfb10, a later checkpoint) on one GPU
cd /root/trusted-sampler; source /root/sampler-venv/bin/activate
export CUDA_VISIBLE_DEVICES=$1
python claims/adv_gen.py --epoch $2 --mode copies
python claims/adv_gen.py --epoch $2 --mode distill --teacher 066dfb10 --samples 16
