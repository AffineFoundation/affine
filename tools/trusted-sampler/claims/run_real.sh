#!/usr/bin/env bash
# one GPU (g in 4..7 -> shard g-4 of 4): real-rollout dots for e62,e64 (refs exist), then ref-only gen for 63,65,66, then dots.
cd /root/trusted-sampler; g=$1; s=$((g-4)); PY=/root/sampler-venv/bin/python
while [ "$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 2000 ]; do sleep 10; done
for e in 62 64; do CUDA_VISIBLE_DEVICES=$g $PY claims/grad_real.py --epoch $e --shard $s --nshards 4; done
sleep $((s*45))
declare -A BASE=([63]=8119f5e2bfe2213c [65]=19f216581a359652 [66]=23a03720b92f9ac7)
for e in 63 65 66; do
  full=$(ls -d /root/ckpt-cache/${BASE[$e]}* | head -1); full=$(basename $full)
  [ -f claims/data/e$e/gen-$s.json ] || CUDA_VISIBLE_DEVICES=$g $PY claims/gen_claims.py --epoch $e --shard $s --nshards 4 --ref-only --base $full
done
for e in 63 65 66; do
  while [ "$(ls claims/data/e$e/gen-[0-9]*.json 2>/dev/null | wc -l)" -lt 4 ]; do sleep 15; done
  CUDA_VISIBLE_DEVICES=$g $PY claims/grad_real.py --epoch $e --shard $s --nshards 4
done
echo ALLDONE
