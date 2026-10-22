#!/usr/bin/env bash
# deadline-trimmed tail: no e66; grads for 63 and 65 after the e65 ref gen finishes
cd /root/trusted-sampler; g=$1; s=$((g-4)); PY=/root/sampler-venv/bin/python
for e in 63 65; do
  while [ "$(ls claims/data/e$e/gen-[0-9]*.json 2>/dev/null | wc -l)" -lt 4 ]; do sleep 10; done
  while [ "$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 2000 ]; do sleep 5; done
  CUDA_VISIBLE_DEVICES=$g $PY claims/grad_real.py --epoch $e --shard $s --nshards 4
done
echo ALLDONE
