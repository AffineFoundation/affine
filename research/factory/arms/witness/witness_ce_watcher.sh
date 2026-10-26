#!/usr/bin/env bash
# WITNESS arm: run action-CE + observation-CE on GPU 6 for each checkpoint
# as the trainer saves it. Appends to /root/work/ce_report.jsonl.
set -u
PY=/root/trainenv/bin/python
export HF_HOME=/dshare/hf
export CUDA_VISIBLE_DEVICES=6

for N in 250 500 750 1000; do
  D=/dshare/ckpt/witness/step$N
  while [ ! -f "$D/adapter_model.safetensors" ]; do
    if ! pgrep -f "distill_sf[t]" >/dev/null; then
      sleep 60  # grace: trainer may have just saved and exited
      if [ ! -f "$D/adapter_model.safetensors" ]; then
        echo "TRAINER_DEAD before step$N"
        exit 1
      fi
      break
    fi
    sleep 120
  done
  sleep 30
  echo "=== evaluating step$N $(date -u +%FT%TZ) ==="
  $PY /root/work/eval_ce.py --lora "$D" \
    --rollouts /root/work/teacher_heldout.jsonl \
    --turns /root/work/turns.jsonl.gz \
    --tag "witness_step${N}_act" --report /root/work/ce_report.jsonl
  $PY /root/work/eval_ce.py --lora "$D" \
    --rollouts /root/work/witness_obs_test.jsonl \
    --turns /root/work/wit_eval_turns.jsonl.gz \
    --tag "witness_step${N}_obs" --report /root/work/ce_report.jsonl
done
echo CE_WATCHER_DONE
