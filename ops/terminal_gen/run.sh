#!/bin/bash
# ops/terminal_gen/run.sh <epoch> [--n-posts N] [--model M] [--base-url U] [--key-env K]
#                                 [--max-usd D] [--concurrency C] [--synth-concurrency S] [--against NAME=DIR]... [--skip-fetch]
#
# Generate a fresh affine_terminal_gen task set for fold epoch <epoch> (seed =
# epoch): Stack Exchange posts -> LLM task specs -> Harbor task dirs -> Docker
# validation -> decontamination against Terminal-Bench 2.0 + 4.0 (+ tmax) ->
# package under rollouts/envs/affine_terminal_gen_v1/affine_terminal_gen_v1/data/e<epoch>/.
#
# WHERE TO RUN: a datagen pod or one rented builder box (Docker + the LLM key
# in the environment). Never the eval pod or the benchmark-fill pods.
# TERMINAL_GEN_OUT must NOT live on the pod's encrypted /root volume
# (gocryptfs/FUSE): validate.py bind-mounts <task>/tests into the container
# and nested docker cannot mount a FUSE path ("change mount propagation
# through procfd ... no such file or directory" on every task, 2026-09-22).
# Default therefore /var/tmp/terminal_gen/out (container overlay).
# Nothing here touches the pods or D; deploying the package is the datagen
# worker's step (ops/king-datagen/deploy_pods.sh) after the handshake in
# the store's internal/benchsuite/requests.md.
#
# Compute + cost for the defaults (300 posts, teacher on Engy): ~$2 of LLM,
# ~300 image builds at 1-5 min each on 4 workers (~2-4 h), ~10 GB of images.
set -euo pipefail
EPOCH=${1:?fold epoch (seed)}; shift
N_POSTS=300; MODEL_ARGS=(); MAX_USD=20; CONC=4; SYNTH_CONC=8; AGAINST=(); SKIP_FETCH=0
while [ $# -gt 0 ]; do
  case "$1" in
    --n-posts) N_POSTS=$2; shift 2;;
    --model|--base-url|--key-env) MODEL_ARGS+=("$1" "$2"); shift 2;;
    --max-usd) MAX_USD=$2; shift 2;;
    --concurrency) CONC=$2; shift 2;;
    --synth-concurrency) SYNTH_CONC=$2; shift 2;;
    --max-tokens|--reasoning-effort) MODEL_ARGS+=("$1" "$2"); shift 2;;   # LLM calls in flight (a spec is ~90 s at the teacher; 8 = 3,000 posts in 10 h, 48 = ~1.5 h)
    --against) AGAINST+=(--against "$2"); shift 2;;
    --skip-fetch) SKIP_FETCH=1; shift;;
    *) echo "unknown arg $1"; exit 2;;
  esac
done
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${TERMINAL_GEN_OUT:-/var/tmp/terminal_gen/out}
PY=${PYTHON:-python}
TMAX_DIR=${TMAX_ROOT:-$HOME/.cache/affine/prime-tasks}/datasets/tmax
[ -d "$TMAX_DIR" ] && AGAINST+=(--against "tmax=$TMAX_DIR")

echo "== epoch $EPOCH: posts ($N_POSTS)"
if [ "$SKIP_FETCH" = 0 ] || [ ! -f "$OUT/e$EPOCH/posts.jsonl.gz" ]; then
  $PY "$HERE/posts.py" --epoch "$EPOCH" --n "$N_POSTS" --out "$OUT"
fi
echo "== synth"
$PY "$HERE/synth.py" --epoch "$EPOCH" --out "$OUT" --max-usd "$MAX_USD" --concurrency "$SYNTH_CONC" "${MODEL_ARGS[@]}"
echo "== render"
$PY "$HERE/harbor.py" --epoch "$EPOCH" --out "$OUT"
echo "== validate (docker)"
$PY "$HERE/validate.py" --epoch "$EPOCH" --out "$OUT" --concurrency "$CONC"
echo "== decontaminate"
$PY "$HERE/decontam.py" --epoch "$EPOCH" --out "$OUT" "${AGAINST[@]}"
echo "== package"
$PY "$HERE/package.py" --epoch "$EPOCH" --out "$OUT"
echo "== summary"
$PY - "$OUT/e$EPOCH" <<'PY'
import json, sys, pathlib
d = pathlib.Path(sys.argv[1])
for f in ("synth_summary.json", "validation_summary.json", "decontam.json"):
    p = d / f
    if p.exists():
        print(f, json.dumps(json.loads(p.read_text())))
PY
echo "done: commit rollouts/envs/affine_terminal_gen_v1/affine_terminal_gen_v1/data/e$EPOCH/, set extra_flags data-epoch $EPOCH and share = 1.0 on [source.affine_terminal_gen], then hand the deploy to the datagen worker (handshake first)"
