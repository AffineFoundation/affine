#!/usr/bin/env bash
cd /root/trusted-sampler; g=$1; PY=/root/sampler-venv/bin/python
while [ "$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 2000 ]; do sleep 5; done
for e in 62 64; do CUDA_VISIBLE_DEVICES=$g $PY claims/plant_teacher_gen.py --epoch $e; done
echo ALLDONE
