#!/bin/bash
# RT-3b live: measure calibration ratio r on fp8 engines + I/II duels (80 turns).
set -euo pipefail
source /root/venv/bin/activate
cd /root
export HF_HOME=/root/hf
export PATH="/usr/local/cuda/bin:$PATH"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p /root/logs /root/results

for port in 8000 8001 8002 8003; do
  fuser -k ${port}/tcp 2>/dev/null || true
done
sleep 2

CUDA_VISIBLE_DEVICES=0,1 nohup vllm serve zai-org/GLM-4.5-Air-FP8 \
  --port 8000 --tensor-parallel-size 2 --max-model-len 32768 \
  --gpu-memory-utilization 0.85 --max-num-batched-tokens 2048 \
  > /root/logs/serve_teacher.log 2>&1 &

python -m harness.serve --kings \
  genesis:8001:2 I:8002:3 II:8003:4

for i in $(seq 1 180); do
  curl -sf -m 2 http://127.0.0.1:8000/v1/models >/dev/null && break
  sleep 10
done
for port in 8001 8002 8003; do
  for i in $(seq 1 120); do
    curl -sf -m 2 http://127.0.0.1:$port/v1/models >/dev/null && break
    sleep 10
  done
done

python -u -m harness.duel \
  --king king-genesis:8001 --challenger king-I:8002 \
  --turns /root/data/turns_minicoder.jsonl --n-turns 80 \
  --ref-cache /root/results/ref_minicoder.jsonl \
  --out /root/results/duel_v2_genesis_I.json \
  --concurrency 16 \
  > /root/logs/duel_v2_I.log 2>&1

python -u -m harness.duel \
  --king king-genesis:8001 --challenger king-II:8003 \
  --turns /root/data/turns_minicoder.jsonl --n-turns 80 \
  --ref-cache /root/results/ref_minicoder.jsonl \
  --out /root/results/duel_v2_genesis_II.json \
  --concurrency 16 \
  > /root/logs/duel_v2_II.log 2>&1

# Emit live calibration ratios from duel jsonl
python - <<'PY'
import json, statistics as st
from pathlib import Path
from harness.score import calibration_ratio
for name in ["duel_v2_genesis_I", "duel_v2_genesis_II"]:
    p = Path(f"/root/results/{name}.jsonl")
    if not p.exists():
        print(name, "missing jsonl"); continue
    by={}
    for line in open(p):
        r=json.loads(line)
        if r.get("valid") and "pairs" in r:
            by.setdefault(r["miner"], []).extend(r["pairs"])
    for m, ps in by.items():
        print(f"{name} {m} r={calibration_ratio(ps):.4f} n_pairs={len(ps)}")
    s=json.load(open(f"/root/results/{name}.json"))
    d=s.get("duel",{})
    print(f"{name} summary z={d.get('z')} margin={d.get('margin')} wins={d.get('challenger_wins')}")
PY
echo RT3_LIVE_DONE
