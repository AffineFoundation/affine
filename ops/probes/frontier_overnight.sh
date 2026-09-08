#!/usr/bin/env bash
# Overnight frontier-rule probe orchestrator (2026-09-06).
#
#   Job A  engy glm-5.2 sampling, records in sequence   -> frontier/<rec>.jsonl + .done
#   Job B  teacher echoes on the probe box, one worker  -> echoes/<rec>.jsonl
#          per record, started as soon as its .done lands
#   Job C  analyze once every echo worker has exited    -> report.{txt,json}
#
# Everything is resumable: re-running this script continues where the jsonl
# files stop. `analyze` can be run by hand at any time on partial data.
#
#   ops/probes/frontier_overnight.sh                 # full run, background
#   RECORDS="chal-00289" LIMIT=20 ops/probes/frontier_overnight.sh   # smoke
#
# Knobs (env): RECORDS, POD (probe box pod name), LIMIT (turns per record),
# ENGY_CONC (default 24), ECHO_CONC (per echo worker, default 32), OUT.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
set -a
# shellcheck disable=SC1091
source .env
# shellcheck disable=SC1091
source "$HOME/.affine-validator.env" 2>/dev/null || true
set +a
# shellcheck disable=SC1091
source .venv/bin/activate

RECORDS="${RECORDS:-chal-00286 chal-00287 chal-00288 chal-00289}"
POD="${POD:-swarm-t-eval-b200-8x-5}"
LIMIT="${LIMIT:-0}"
ENGY_CONC="${ENGY_CONC:-24}"
ECHO_CONC="${ECHO_CONC:-32}"
OUT="${OUT:-$REPO/research/results/frontier_rule_probe}"
LOGS="$OUT/logs"
mkdir -p "$LOGS"
PROBE="python research/scripts/frontier_rule_probe.py --out $OUT"
STATUS="$OUT/STATUS"

stamp() { date -u +%FT%TZ; }
note() { echo "[$(stamp)] $*" | tee -a "$LOGS/orchestrator.log"; }
status() { echo "$(stamp) $*" > "$STATUS"; }

for rec in $RECORDS; do
  [[ -s "$OUT/turns/$rec.jsonl" ]] || { note "no turns for $rec — run select first"; exit 1; }
done

note "start records=[$RECORDS] pod=$POD limit=$LIMIT engy_conc=$ENGY_CONC echo_conc=$ECHO_CONC"
status "running"

# Job A: sequential sampling (one engy stream; .done per record).
(
  for rec in $RECORDS; do
    if [[ -f "$OUT/frontier/$rec.done" && "$LIMIT" == "0" ]]; then
      note "A $rec already done"; continue
    fi
    note "A $rec sampling"
    $PROBE sample --record "$rec" --concurrency "$ENGY_CONC" --limit "$LIMIT" \
      >> "$LOGS/sample.$rec.log" 2>&1 || note "A $rec FAILED (see logs/sample.$rec.log)"
    [[ "$LIMIT" == "0" ]] || touch "$OUT/frontier/$rec.done"   # smoke: mark partial as done
    note "A $rec finished"
  done
  note "A all records finished"
) &
A_PID=$!

# Job B: one echo worker per record, launched when its frontier file is done.
B_PIDS=()
for rec in $RECORDS; do
  (
    while [[ ! -f "$OUT/frontier/$rec.done" ]]; do sleep 30; done
    note "B $rec echo start"
    $PROBE echo --record "$rec" --pod "$POD" --concurrency "$ECHO_CONC" --limit "$LIMIT" \
      >> "$LOGS/echo.$rec.log" 2>&1 || note "B $rec FAILED (see logs/echo.$rec.log)"
    note "B $rec echo finished"
  ) &
  B_PIDS+=($!)
done

wait "$A_PID"
for p in "${B_PIDS[@]}"; do wait "$p"; done

# Job C
note "C analyze"
if $PROBE analyze > "$LOGS/analyze.log" 2>&1; then
  note "C done -> $OUT/report.txt"
  status "done"
else
  note "C FAILED (see logs/analyze.log)"
  status "analyze failed"
fi
