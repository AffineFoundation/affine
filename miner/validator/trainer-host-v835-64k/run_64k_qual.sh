#!/usr/bin/env bash
# SN120 64K H200 qualification for trainer-host-v835-64k (ZeRO-2 grad sharding).
#
# Intended host: rented 8xH200 (≈141 GB / ≈140 GiB allocator-visible each).
# This machine may have no GPUs during authoring; the script fails fast if CUDA
# is missing when a GPU step is requested.
#
# What it does:
#   1) CPU/1-2GPU numerical equivalence (v834c accumulator vs v835 sharded)
#   2) Static memory estimate JSON
#   3) Matched 32K synthetic step for v834c and v835
#   4) 64K synthetic step for v834c (expect OOM) and v835 (expect success)
#   5) Writes per-GPU peak memory + step times under $OUT_DIR
#
# Dependencies (install on the H200 box before running):
#   - Python 3.11+ with CUDA torch (live stack used torch 2.14 / 2.15)
#   - transformers 5.14.x providing Qwen3_5ForConditionalGeneration
#   - flash-attn (or torch.nn.attention.varlen FlashAttention-2)
#   - flash-linear-attention / fla 0.6.x (causal_conv1d, chunk_gated_delta_rule)
#   - safetensors, accelerate (transformers deps)
#
# Model:
#   Default: random://qwen3_5_9b  (same config as live 9B hybrid, random BF16 init)
#   Or set MODEL to a public HF id / local path of the same architecture, e.g.
#     export MODEL=Qwen/Qwen3.5-9B   # if/when that public checkpoint exists
#   Do NOT point at private R2 / wallet material.
#
# Usage:
#   cd miner/validator/trainer-host-v835-64k
#   ./run_64k_qual.sh
#   # optional:
#   OUT_DIR=/tmp/sn120-64k-qual NPROC=8 MODEL=random://qwen3_5_9b ./run_64k_qual.sh
#   ACTIVATION_OFFLOAD=1 ./run_64k_qual.sh          # if 64K still tight
#   SKIP_V834C_64K=1 ./run_64k_qual.sh              # skip the expected-OOM baseline
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/../../../.venv/bin/activate" ]]; then
  # repo-root venv when present
  # shellcheck disable=SC1091
  source "${ROOT}/../../../.venv/bin/activate" || true
elif [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/.venv/bin/activate"
fi

OUT_DIR="${OUT_DIR:-$ROOT/qual-results}"
NPROC="${NPROC:-8}"
MODEL="${MODEL:-random://qwen3_5_9b}"
PROMPT_LEN="${PROMPT_LEN:-4096}"
LOGIT_CHUNK="${LOGIT_CHUNK:-4096}"
ACTIVATION_OFFLOAD="${ACTIVATION_OFFLOAD:-0}"
SKIP_V834C_64K="${SKIP_V834C_64K:-0}"
PYTHON="${PYTHON:-python}"

mkdir -p "$OUT_DIR"
echo "[qual] root=$ROOT out=$OUT_DIR nproc=$NPROC model=$MODEL"

echo "[qual] (1/5) memory estimate"
"$PYTHON" memory_estimate.py --json-out "$OUT_DIR/memory_estimate.json"

echo "[qual] (2/5) numerical equivalence (CPU)"
"$PYTHON" numerical_equivalence_test.py --device cpu --json-out "$OUT_DIR/equiv_cpu.json" \
  >"$OUT_DIR/equiv_cpu.stdout.json"
"$PYTHON" -c 'import json,sys; r=json.load(open(sys.argv[1])); print("[qual] equiv_cpu ok=", r.get("ok")); sys.exit(0 if r.get("ok") else 1)' \
  "$OUT_DIR/equiv_cpu.json"

if command -v torchrun >/dev/null 2>&1; then
  echo "[qual] (2b) numerical equivalence (2-proc CPU gloo, if torchrun works)"
  torchrun --standalone --nproc_per_node=2 \
    numerical_equivalence_test.py --device cpu --json-out "$OUT_DIR/equiv_2proc_cpu.json" \
    >"$OUT_DIR/equiv_2proc_cpu.stdout.json" \
    || echo "[qual] 2-proc CPU equivalence skipped/failed (non-fatal for GPU qual)"
fi

if ! "$PYTHON" - <<'PY'
import torch, sys
sys.exit(0 if torch.cuda.is_available() and torch.cuda.device_count() >= 1 else 1)
PY
then
  echo "[qual] No CUDA GPUs visible — wrote CPU equivalence + memory estimate only."
  echo "[qual] Re-run on the 8xH200 box to execute steps 3–5."
  exit 0
fi

GPU_COUNT="$("$PYTHON" - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
if [[ "$GPU_COUNT" -lt "$NPROC" ]]; then
  echo "[qual] warning: only $GPU_COUNT GPUs, adjusting NPROC"
  NPROC="$GPU_COUNT"
fi

OFFLOAD_FLAG=()
if [[ "$ACTIVATION_OFFLOAD" == "1" ]]; then
  OFFLOAD_FLAG=(--activation-offload)
fi

run_step() {
  local variant="$1"
  local out_len="$2"
  local tag="$3"
  local out_json="$OUT_DIR/${tag}.json"
  echo "[qual] torchrun variant=$variant output_len=$out_len -> $out_json"
  # Allow OOM to be recorded in JSON instead of aborting the whole script.
  set +e
  torchrun --standalone --nproc_per_node="$NPROC" \
    qual_step.py \
      --variant "$variant" \
      --output-len "$out_len" \
      --prompt-len "$PROMPT_LEN" \
      --model "$MODEL" \
      --logit-chunk "$LOGIT_CHUNK" \
      --json-out "$out_json" \
      "${OFFLOAD_FLAG[@]}"
  local rc=$?
  set -e
  echo "[qual] finished tag=$tag rc=$rc"
  return 0
}

echo "[qual] (3/5) matched 32K comparison (output_len=32768, context=36864)"
run_step v834c 32768 step_v834c_32k
run_step v835 32768 step_v835_32k

echo "[qual] (4/5) 64K v834c baseline (output_len=65536, context=69632) — expect OOM"
if [[ "$SKIP_V834C_64K" == "1" ]]; then
  echo "[qual] skipped v834c 64K by request"
else
  run_step v834c 65536 step_v834c_64k
fi

echo "[qual] (5/5) 64K v835 candidate"
run_step v835 65536 step_v835_64k

"$PYTHON" - <<PY
import json, pathlib
out = pathlib.Path("$OUT_DIR")
summary = {"out_dir": str(out), "files": sorted(p.name for p in out.glob("*.json"))}
for name in ["step_v834c_32k.json", "step_v835_32k.json", "step_v834c_64k.json", "step_v835_64k.json"]:
    path = out / name
    if not path.exists():
        summary[name] = None
        continue
    data = json.loads(path.read_text())
    summary[name] = {
        "any_oom": data.get("any_oom"),
        "max_allocated_gib": (data.get("max_allocated_bytes") or 0) / 1024**3,
        "max_reserved_gib": (data.get("max_reserved_bytes") or 0) / 1024**3,
        "world": data.get("world"),
        "output_len": data.get("output_len"),
    }
(out / "SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo "[qual] done. Send $OUT_DIR/SUMMARY.json and the step_*.json files back for iteration."
