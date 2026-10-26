#!/usr/bin/env bash
# WORLDMODEL arm milestone: CE evals (GPU 6) + merge/ship for one checkpoint.
# usage: wm_milestone.sh STEPNUM
set -uo pipefail
N="$1"
CK="/dshare/ckpt/worldmodel/step${N}"
PY=/root/trainenv/bin/python
export HF_HOME=/dshare/hf

(setsid nohup bash -c "
  CUDA_VISIBLE_DEVICES=6 $PY /root/work/eval_ce.py \
    --lora $CK \
    --rollouts /root/work/witness_obs_test.jsonl \
    --turns /root/work/wm_eval_turns.jsonl.gz \
    --tag wm_step${N}_obs --report /root/work/ce_report.jsonl &&
  CUDA_VISIBLE_DEVICES=6 $PY /root/work/eval_ce.py \
    --lora $CK \
    --rollouts /root/work/teacher_heldout.jsonl \
    --turns /root/work/turns.jsonl.gz \
    --tag wm_step${N}_act --report /root/work/ce_report.jsonl
" > /root/work/eval_wm_${N}.log 2>&1 < /dev/null &)

(setsid nohup env BASE_MODEL=Qwen/Qwen3-8B \
  bash /root/work/merge_ship.sh worldmodel "$CK" \
  > /root/work/ship_wm_${N}.log 2>&1 < /dev/null &)

sleep 3
pgrep -af "eval_c[e]" | head -2
pgrep -af "merge_shi[p]" | head -2
echo "MILESTONE_${N}_LAUNCHED"
