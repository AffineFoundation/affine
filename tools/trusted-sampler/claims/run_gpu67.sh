#!/bin/bash
cd /root/trusted-sampler; source /root/sampler-venv/bin/activate
python claims/adv_scratch_spec.py 64 copies:hint,distill > logs/adv-spec2.log 2>&1
CUDA_VISIBLE_DEVICES=6 setsid nohup python claims/adv_scratch.py --epoch 64 --ids adv_copies_hint-1,adv_distill-0,adv_distill-1 --steps 2 --modes sgd --tag b > logs/adv-scratch-e64-g6b.log 2>&1 < /dev/null &
CUDA_VISIBLE_DEVICES=7 setsid nohup python claims/adv_band.py --epoch 64 > logs/adv-band-e64-g7.log 2>&1 < /dev/null &
