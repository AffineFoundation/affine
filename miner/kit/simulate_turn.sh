#!/usr/bin/env bash
# End-to-end local turn: head -> rollouts -> batch file -> the live trainer step -> held-out gate decision.
# Needs a GPU box with vLLM 0.32.0 (requirements-gate-vllm.txt) for sampling and requirements-trainer.txt for the step
# (the 9B step needs ~80 GB+ in one process; the live trainer uses 8 GPUs). Nothing here uploads anything.
#   HOTKEY=5F... bash miner/kit/simulate_turn.sh [WORK_DIR]
# Env knobs: TASKS (default 0-7495), K (8), N_GATE (expected draw size, live prior-v1 N 500), PORT_H (8000), PORT_C (8001), SELECT (correct1), LR / KL (unset = v1).
set -euo pipefail
KIT="$(cd "$(dirname "$0")" && pwd)"; W="${1:-./turn}"; mkdir -p "$W"
: "${HOTKEY:?set HOTKEY to your registered ss58 hotkey}"
TASKS="${TASKS:-0-7495}"; K="${K:-8}"; N_GATE="${N_GATE:-500}"; PORT_H="${PORT_H:-8000}"; PORT_C="${PORT_C:-8001}"; SELECT="${SELECT:-correct1}"
SERVE=(--served-model-name m --dtype bfloat16 --max-model-len 16384 --enable-prefix-caching --max-num-seqs 512)   # = validator/probe/serve.sh
wait_up() { for _ in $(seq 600); do curl -sf "http://127.0.0.1:$1/v1/models" >/dev/null && return 0; sleep 2; done; echo "vLLM on :$1 did not start" >&2; exit 1; }

echo "== 1. head + KL reference (sha256-verified)"
python3 "$KIT/fetch_head.py" --out "$W/head"
[ -f "$W/ref/CHECKPOINT.json" ] || python3 "$KIT/fetch_head.py" --out "$W/ref" --reference

echo "== 2. sample rollouts from the head (gate settings, trainer prompt rendering)"
vllm serve "$W/head" --port "$PORT_H" "${SERVE[@]}" > "$W/vllm-head.log" 2>&1 & VH=$!
trap 'kill $VH ${VC:-} 2>/dev/null || true' EXIT
wait_up "$PORT_H"
python3 "$KIT/generate.py" --head "$W/head" --url "http://127.0.0.1:$PORT_H" --tasks "$TASKS" --k "$K" --out "$W/rollouts.jsonl"

echo "== 3. batch file + the validator's full check"
EXTRA=(); [ -n "${LR:-}" ] && EXTRA+=(--lr "$LR"); [ -n "${KL:-}" ] && EXTRA+=(--kl "$KL")
python3 "$KIT/make_batch.py" --head "$W/head" --hotkey "$HOTKEY" --rollouts "$W/rollouts.jsonl" --out "$W/batch.json" --select "$SELECT" "${EXTRA[@]}"
python3 "$KIT/check_batch.py" --hotkey "$HOTKEY" --tokenizer "$W/head" "$W/batch.json"

echo "== 4. one v8.3.3l step (live trainer code) -> candidate"
kill $VH; wait $VH 2>/dev/null || true
python3 "$KIT/train_step.py" --head "$W/head" --ref "$W/ref" --batch "$W/batch.json" --hotkey "$HOTKEY" --out "$W/cand" --device cuda --report "$W/step.json"

echo "== 5. held-out gate: head vs candidate"
vllm serve "$W/head" --port "$PORT_H" "${SERVE[@]}" > "$W/vllm-head.log" 2>&1 & VH=$!
vllm serve "$W/cand" --port "$PORT_C" "${SERVE[@]}" > "$W/vllm-cand.log" 2>&1 & VC=$!
wait_up "$PORT_H"; wait_up "$PORT_C"
python3 "$KIT/gate_eval.py" --head "$W/head" --head-url "http://127.0.0.1:$PORT_H" --cand-url "http://127.0.0.1:$PORT_C" --n "$N_GATE" --out "$W/gate.json"
echo "done: $W/batch.json is upload-ready (python3 miner/affine_batch.py upload --hotkey \$HOTKEY --key <seed file> $W/batch.json; see miner/README.md)"
