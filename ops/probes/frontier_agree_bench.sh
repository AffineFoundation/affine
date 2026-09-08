#!/usr/bin/env bash
# Frontier-agreement vs bench (2026-09-07): sample engy glm-5.2 on every selected
# record, PAR records at a time, then analyze. Resumable (per-record .done).
set -u
cd "$(dirname "$0")/../.."
source .env; source .venv/bin/activate
OUT=research/results/frontier_agree_bench
PAR=${PAR:-4}
mkdir -p "$OUT/logs" "$OUT/frontier"
ls "$OUT/turns" | sed 's/\.jsonl$//' | while read -r rec; do
  [[ -f "$OUT/frontier/$rec.done" ]] || echo "$rec"
done | xargs -P "$PAR" -I{} bash -c \
  'python research/scripts/frontier_agree_bench.py sample --record {} --concurrency 12 > research/results/frontier_agree_bench/logs/{}.log 2>&1; echo "$(date -u +%FT%TZ) {} exit $?"'
python research/scripts/frontier_agree_bench.py analyze > "$OUT/logs/analyze.log" 2>&1
echo "$(date -u +%FT%TZ) analyze exit $?"
