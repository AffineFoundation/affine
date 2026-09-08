#!/bin/bash
# Duel-shaped throughput sweep against the eval-tune box (not the live router).
# Usage: ./evalopt_sweep.sh <tag> <concurrency,list> [n_turns]
set -euo pipefail
cd /home/const/subnet120
source .venv/bin/activate
source ops/teacher-swarm/.swarm_env
TAG=${1:?tag}
CONCS=${2:-8,16,24,36,48,64}
TURNS=${3:-48}
URLS_FILE=ops/teacher-swarm/state/evalopt_urls.txt
OUTDIR=ops/teacher-swarm/state/evalopt
mkdir -p "$OUTDIR"
if [ ! -s "$URLS_FILE" ]; then
  echo "missing $URLS_FILE — write one base URL per line" >&2
  exit 2
fi
mapfile -t URLS < "$URLS_FILE"
ARGS=()
for u in "${URLS[@]}"; do
  [ -n "$u" ] && ARGS+=(--base-url "$u")
done
MODEL=${MODEL:-Qwen/Qwen3.8-27B}
echo "tag=$TAG urls=${#ARGS[@]} concs=$CONCS turns=$TURNS model=$MODEL"
IFS=',' read -ra CS <<< "$CONCS"
for c in "${CS[@]}"; do
  out="$OUTDIR/${TAG}_c${c}.json"
  echo "=== $TAG c=$c ==="
  python ops/teacher-b200/bench_duel_real.py \
    "${ARGS[@]}" \
    --model "$MODEL" \
    --api-key "$SWARM_KEY" \
    --concurrency "$c" \
    --n-turns "$TURNS" \
    --prefix-tokens 8192 \
    --suffix-tokens 1024 \
    --echoes-per-turn 8 \
    --sample-max-tokens 512 \
    --retries 2 \
    --timeout-s 900 \
    --out "$out"
done
python - <<PY
import json, pathlib
p = pathlib.Path("ops/teacher-swarm/state/evalopt")
rows = []
for f in sorted(p.glob("${TAG}_c*.json")):
    d = json.loads(f.read_text())
    rows.append((d["concurrency"], d["turns_per_s"], d["echo"]["p50_s"],
                 d["echo"]["p95_s"], d["wall_s"], d.get("retries", 0)))
print(f"{'c':>4} {'tps':>7} {'echo_p50':>9} {'echo_p95':>9} {'wall':>7} {'retries':>7}")
for r in rows:
    print(f"{r[0]:4d} {r[1]:7.3f} {r[2]:9.3f} {r[3]:9.3f} {r[4]:7.1f} {r[5]:7d}")
best = max(rows, key=lambda x: x[1]) if rows else None
if best:
    print(f"BEST c={best[0]} tps={best[1]}")
    pathlib.Path("ops/teacher-swarm/state/evalopt_best_${TAG}.txt").write_text(
        f"c={best[0]} tps={best[1]} echo_p50={best[2]} echo_p95={best[3]}\n")
PY
