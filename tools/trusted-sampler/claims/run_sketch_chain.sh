#!/bin/bash
cd /root/trusted-sampler; source /root/sampler-venv/bin/activate; export CUDA_VISIBLE_DEVICES=4
while pgrep -f "adv_dots.py --epoch 64 --name distill" > /dev/null; do sleep 15; done
sleep 90
for E in 64 62 66; do python claims/adv_sketch.py --epoch $E > logs/adv-sketch-e$E-g4.log 2>&1; done
