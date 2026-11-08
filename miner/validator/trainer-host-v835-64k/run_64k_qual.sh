#!/usr/bin/env bash
# SN120 64K H200 qualification for trainer-host-v835-64k (async ZeRO-2 grad sharding).
#
# Intended host: rented 8xH200 (≈141 GB / ≈140 GiB allocator-visible each).
#
# What it does:
#   1) CPU / 2-proc numerical equivalence (multi-micro-batch + shard-clip)
#   2) Static memory estimate JSON
#   3) Matched 32K singleton step for v834c and v835
#   4) 64K singleton step for v834c and v835
#   5) Packed 125×8 synthetic batch (several micro-batches) for v835
#   6) Fails if any rank reports an exception (not only OOM)
#
# Dependencies (install on the H200 box before running):
#   - Python 3.11+ with CUDA torch (live stack used torch 2.14 / 2.15)
#   - transformers 5.14.x providing Qwen3_5ForConditionalGeneration
#   - flash-attn (or torch.nn.attention.varlen FlashAttention-2)
#   - flash-linear-attention / fla 0.6.x
#   - triton >= 3.7.1   # 3.7.0 breaks fla gated chunk_bwd on Hopper
#   - safetensors, accelerate
#
# Model:
#   Default: random://qwen3_5_9b
#   Or: export MODEL=/path/to/public-qwen3.5-9b
#
# Usage:
#   cd miner/validator/trainer-host-v835-64k
#   ./run_64k_qual.sh
#   OUT_DIR=/tmp/sn120-64k-qual NPROC=8 ./run_64k_qual.sh
#   SKIP_V834C_64K=1 ./run_64k_qual.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/../../../.venv/bin/activate" ]]; then
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
SKIP_V834C_64K="${SKIP_V834C_64K:-0}"
SKIP_PACKED="${SKIP_PACKED:-0}"
PYTHON="${PYTHON:-python}"
export TRITON_CACHE_ROOT="${TRITON_CACHE_ROOT:-/tmp/triton-cache-v835}"

mkdir -p "$OUT_DIR" "$TRITON_CACHE_ROOT"
echo "[qual] root=$ROOT out=$OUT_DIR nproc=$NPROC model=$MODEL"

echo "[qual] checking triton>=3.7.1 (Hopper fla)"
"$PYTHON" - <<'PY'
import sys
try:
    import triton
except ImportError:
    print("[qual] triton not importable yet (ok if GPU stack installs later)"); sys.exit(0)
ver = getattr(triton, "__version__", "0")
parts = []
for x in ver.split(".")[:3]:
    try: parts.append(int("".join(ch for ch in x if ch.isdigit()) or "0"))
    except Exception: parts.append(0)
while len(parts) < 3: parts.append(0)
if tuple(parts) < (3, 7, 1):
    print(f"[qual] FATAL: triton>=3.7.1 required for fla gated chunk_bwd on Hopper; found {ver}", file=sys.stderr)
    sys.exit(1)
print("[qual] triton", ver, "ok")
PY

echo "[qual] (1/6) memory estimate"
"$PYTHON" memory_estimate.py --json-out "$OUT_DIR/memory_estimate.json"

echo "[qual] (2/6) numerical equivalence (CPU, multi-micro-batch + shard-clip)"
"$PYTHON" numerical_equivalence_test.py --device cpu --json-out "$OUT_DIR/equiv_cpu.json" \
  >"$OUT_DIR/equiv_cpu.stdout.json"
"$PYTHON" -c 'import json,sys; r=json.load(open(sys.argv[1])); print("[qual] equiv_cpu ok=", r.get("ok"), "shard_clip=", r.get("shard_clip")); sys.exit(0 if r.get("ok") else 1)' \
  "$OUT_DIR/equiv_cpu.json"

if command -v torchrun >/dev/null 2>&1; then
  echo "[qual] (2b) numerical equivalence (2-proc CPU)"
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
  echo "[qual] Re-run on the 8xH200 box to execute GPU steps."
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

check_json() {
  local path="$1"
  "$PYTHON" - <<PY
import json, sys
p = "$path"
d = json.load(open(p))
err = d.get("any_error") or any(r.get("error") for r in d.get("ranks") or [])
oom = d.get("any_oom")
print(f"[qual] {p}: any_oom={oom} any_error={err} max_alloc_GiB={(d.get('max_allocated_bytes') or 0)/1024**3:.2f}")
if err:
    print("[qual] FATAL: rank exception(s):", [r.get("error") for r in d.get("ranks") or [] if r.get("error")], file=sys.stderr)
    sys.exit(2)
PY
}

run_step() {
  local variant="$1"
  local mode="$2"
  local out_len="$3"
  local tag="$4"
  local out_json="$OUT_DIR/${tag}.json"
  echo "[qual] torchrun variant=$variant mode=$mode output_len=$out_len -> $out_json"
  set +e
  torchrun --standalone --nproc_per_node="$NPROC" \
    qual_step.py \
      --variant "$variant" \
      --mode "$mode" \
      --output-len "$out_len" \
      --prompt-len "$PROMPT_LEN" \
      --model "$MODEL" \
      --logit-chunk "$LOGIT_CHUNK" \
      --json-out "$out_json"
  local rc=$?
  set -e
  echo "[qual] finished tag=$tag rc=$rc"
  if [[ -f "$out_json" ]]; then
    check_json "$out_json" || return $?
  elif [[ "$rc" -ne 0 ]]; then
    echo "[qual] FATAL: $tag produced no JSON (rc=$rc)" >&2
    return "$rc"
  fi
  return 0
}

echo "[qual] (3/6) matched 32K singleton (output_len=32768)"
run_step v834c singleton 32768 step_v834c_32k
run_step v835 singleton 32768 step_v835_32k

echo "[qual] (4/6) 64K singleton (output_len=65536)"
if [[ "$SKIP_V834C_64K" == "1" ]]; then
  echo "[qual] skipped v834c 64K by request"
else
  # v834c may OOM (recorded in JSON) or succeed near the ceiling; do not abort the suite on OOM alone.
  set +e
  torchrun --standalone --nproc_per_node="$NPROC" \
    qual_step.py --variant v834c --mode singleton --output-len 65536 --prompt-len "$PROMPT_LEN" \
    --model "$MODEL" --logit-chunk "$LOGIT_CHUNK" --json-out "$OUT_DIR/step_v834c_64k.json"
  set -e
  if [[ -f "$OUT_DIR/step_v834c_64k.json" ]]; then
    "$PYTHON" - "$OUT_DIR/step_v834c_64k.json" <<'PY'
import json, sys
p = sys.argv[1]
d = json.load(open(p))
print("[qual]", p, "any_oom=", d.get("any_oom"), "any_error=", d.get("any_error"),
      "max_alloc_GiB=", (d.get("max_allocated_bytes") or 0) / 1024**3)
errs = [r.get("error") for r in d.get("ranks") or [] if r.get("error") and not r.get("oom")]
if errs:
    raise SystemExit("v834c 64K non-OOM errors: " + repr(errs))
PY
  fi
fi
run_step v835 singleton 65536 step_v835_64k

echo "[qual] (5/6) packed 125x8 multi-microbatch (v835)"
if [[ "$SKIP_PACKED" == "1" ]]; then
  echo "[qual] skipped packed batch"
else
  set +e
  torchrun --standalone --nproc_per_node="$NPROC" \
    qual_step.py --variant v835 --mode packed --output-len 32768 --prompt-len 512 \
    --model "$MODEL" --logit-chunk "$LOGIT_CHUNK" --group-count 125 --group-size 8 \
    --token-budget 69632 --json-out "$OUT_DIR/step_v835_packed125x8.json"
  local_rc=$?
  set -e
  if [[ -f "$OUT_DIR/step_v835_packed125x8.json" ]]; then
    check_json "$OUT_DIR/step_v835_packed125x8.json"
  else
    echo "[qual] FATAL: packed run produced no JSON (rc=$local_rc)" >&2
    exit "$local_rc"
  fi
fi

echo "[qual] (6/6) summary"
OUT_DIR="$OUT_DIR" "$PYTHON" - <<'PY'
import json, os, pathlib
out = pathlib.Path(os.environ["OUT_DIR"])
summary = {"out_dir": str(out), "files": sorted(p.name for p in out.glob("*.json"))}
for name in ["step_v834c_32k.json", "step_v835_32k.json", "step_v834c_64k.json",
             "step_v835_64k.json", "step_v835_packed125x8.json", "equiv_cpu.json"]:
    path = out / name
    if not path.exists():
        summary[name] = None
        continue
    data = json.loads(path.read_text())
    if name.startswith("equiv"):
        summary[name] = {"ok": data.get("ok"), "shard_clip": data.get("shard_clip")}
    else:
        summary[name] = {
            "any_oom": data.get("any_oom"),
            "any_error": data.get("any_error"),
            "max_allocated_gib": (data.get("max_allocated_bytes") or 0) / 1024**3,
            "max_reserved_gib": (data.get("max_reserved_bytes") or 0) / 1024**3,
            "world": data.get("world"),
            "mode": data.get("mode"),
            "output_len": data.get("output_len"),
        }
(out / "SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo "[qual] done. Send $OUT_DIR/SUMMARY.json and the step_*.json files back for iteration."
