#!/usr/bin/env bash
# End-to-end smoke test of the loop on GPUs 6-7 (tiny sizes, 2 rounds).
# Exercises: G serving + LoRA hot-swap, sampling, D scoring/training, g_sft,
# metrics, rollback bookkeeping. Runs while teacher gen owns GPUs 0-5.
set -uo pipefail
S=/dshare/smoke
mkdir -p "$S"
PY=/root/trainenv/bin/python
export HF_HOME=/dshare/hf
export HF_TOKEN="$(cat /root/.hf_token)"
log() { echo "[smoke $(date -u +%T)] $*"; }

log "waiting for a teacher replica (port 8010)"
i=0
while ! curl -sf http://127.0.0.1:8010/v1/models >/dev/null 2>&1; do
  sleep 15; i=$((i+15)); [ "$i" -ge 2400 ] && { log "teacher never healthy"; exit 1; }
done

log "smoke G server on GPU 6 (port 8601)"
bash /root/work/serve_h200.sh Qwen/Qwen3-4B 8601 6 \
  --enable-lora --max-lora-rank 16 --max-loras 4
i=0
while ! curl -sf http://127.0.0.1:8601/v1/models >/dev/null 2>&1; do
  sleep 10; i=$((i+10)); [ "$i" -ge 900 ] && { log "G server never healthy"; exit 1; }
done

log "tiny teacher caches"
$PY /root/work/sample_rollouts.py --urls http://127.0.0.1:8010 \
  --model Qwen/Qwen3-32B --tokenizer Qwen/Qwen3-32B \
  --split train --k 2 --limit 8 --workers 8 --out "$S/teacher_train.jsonl"
$PY /root/work/sample_rollouts.py --urls http://127.0.0.1:8010 \
  --model Qwen/Qwen3-32B --tokenizer Qwen/Qwen3-32B \
  --split test --k 2 --limit 6 --seed 1 --workers 8 --out "$S/teacher_heldout.jsonl"

log "driver: 2 tiny rounds on GPU 7"
CUDA_VISIBLE_DEVICES=7 $PY /root/work/gad_driver.py \
  --teacher-train "$S/teacher_train.jsonl" \
  --teacher-heldout "$S/teacher_heldout.jsonl" \
  --g-url http://127.0.0.1:8601 \
  --sft-gpu 7 --rounds 2 --batch-turns 6 --n-cands 3 \
  --heldout-n 4 --heldout-cands 2 --d-steps 6 --g-steps 6 \
  --work "$S" 2>&1 | tail -40

docker rm -f vllm_8601 >/dev/null 2>&1 || true
log "SMOKE_DONE rc=$?"
cat "$S/status.log" 2>/dev/null
