#!/bin/bash
# after adv_gen (distill) finishes on GPU4: distill dots -> band test -> sign-mode (Adam-like) scratch on 3 batches
cd /root/trusted-sampler; source /root/sampler-venv/bin/activate; export CUDA_VISIBLE_DEVICES=4
while pgrep -f "adv_gen.py --epoch 64" > /dev/null; do sleep 15; done
python claims/adv_dots.py --epoch 64 --name distill > logs/adv-dots-distill-g4.log 2>&1
python claims/adv_band.py --epoch 64 > logs/adv-band-e64-g4.log 2>&1
python claims/adv_scratch.py --epoch 64 --ids beyond_policy-0,honest-0,length_x32-0 --steps 1 --modes sign,sgd --shard 0 --nshards 1 --tag sign > logs/adv-scratch-sign-e64-g4.log 2>&1
