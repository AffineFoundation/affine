#!/bin/bash
# "Genesis (vendor settings)" reference row (Alan / Jacob 2026-09-19): the chat cells and
# Terminal-Bench 2 at the Qwen3.6-35B-A3B model card's settings instead of the suite's
# fixed budget — T 1.0 / top_p 0.95 / top_k 20 / presence_penalty 1.5 / 81,920 tokens per
# call; TB2 at 3 h per task, 2 attempts averaged, 8 vCPU / 16 GB sandboxes (the card used
# 32 / 48; that would cap Daytona at 15 in flight under our 1000 GB quota). Published as a
# separate reference card (digest hf-995ad96eac-vendor, label genesis-vendor), never on the
# Genesis row itself. Queued behind the current fast passes (waits for the given run's SWE
# Daytona job so the memory quota is free).
#   bash vendor_row.sh [wait_for_run_id]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
WAIT="${1:-}"
if [ -n "$WAIT" ]; then
  while ! grep -q "done king/swebench-verified\|FAIL swebench-verified" "$HOME/benchsuite/runs/$WAIT/swe-box.log" 2>/dev/null; do sleep 300; done
fi
RUN="$(date -u +%Y%m%dT%H%MZ)-genesis-vendor"
export FAST_GROUPS="chat,tb2"
export BENCHSUITE_SETTINGS_JSON='{"temperature": 1.0, "max_tokens": 81920, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5}'
export BENCHSUITE_SETTINGS_NOTE="Qwen/Qwen3.6-35B-A3B model card: T 1.0, top_p 0.95, top_k 20, presence_penalty 1.5, max_tokens 81920; TB2 3 h/task, 2 attempts, 8 vCPU / 16 GB sandboxes (card: 32 / 48)"
export BENCHSUITE_REFERENCE_ROW="Genesis (vendor settings)"
export BENCHSUITE_DIGEST_SUFFIX="-vendor"
export FAST_TB2_TIMEOUT_S=10800 FAST_TB2_IN_FLIGHT=60
export FAST_TB2_ARGS="--budget-tag vendor --temperature 1.0 --attempts 2 --sandbox-cpus 8 --sandbox-mem-mb 16384"
echo "[vendor-row] $(date -u +%FT%TZ) start $RUN"
bash pass.sh "hf://Qwen/Qwen3.6-35B-A3B@995ad96eacd98c81ed38be0c5b274b04031597b0" genesis-vendor "$RUN" fast
echo "[vendor-row] $(date -u +%FT%TZ) $RUN exit=$(cat "state/pass-$RUN.exit" 2>/dev/null)"
