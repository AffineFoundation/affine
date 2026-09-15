#!/bin/bash
# Agentic-set backfill queue (operator order 2026-09-15):
#   reign 13 -> teacher -> genesis -> uid 69 -> reign 11 for the standard agentic
#   cells (modes.agentic_std_envs), then SWE-bench Pro for the current king and the
#   teacher, one model at a time, only when Lium has spare 1x H200 stock (the coverage
#   backfill's demand comes first).
# Each job is one `agentic` pass merged into the model's card; a job whose rent fails
# for lack of stock (exit 2) is retried every 10 min. Two workers run the standard
# jobs; SWE-bench Pro runs alone after them.
#   bash agentic_queue.sh            # start both workers (idempotent: skips jobs whose cells exist)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="${BENCHSUITE_PYTHON:-$REPO/.venv/bin/python}"
BENCH_HOME="${BENCH_HOME:-$HOME/benchsuite}"
cd "$HERE"
STD=$("$PY" -c 'import tomllib; print(",".join(tomllib.load(open("suite.toml","rb"))["modes"]["agentic_std_envs"]))')
log() { echo "[agentic-queue] $(date -u +%FT%TZ) $*"; }

R11=0ce59769300c234203dcaa786e50f3a23b564bb65e83db0a711a7a43c7951952
R13=6d0ee567e33ee44fc238cc4a1eaae3ad9c98e3e63669548bc2fa75b94c5ebb47
U69=6c877fd2242df22b0aebf28568821a16cb7d56f5ce5bb29d3936172b48943e2e
GEN="hf://Qwen/Qwen3.6-35B-A3B@995ad96eacd98c81ed38be0c5b274b04031597b0"
TEA="hf://Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"

# missing_cells <run_dir> <as> <envs csv> -> csv of envs without a summary.json in that card
missing_cells() {
  local out=""
  IFS=, read -ra ENVS <<< "$3"
  for e in "${ENVS[@]}"; do
    [ -f "$BENCH_HOME/runs/$1/$2/${e}__t0/summary.json" ] || out="${out:+$out,}$e"
  done
  echo "$out"
}

# job <ref> <label> <merge_into> <merge_as> <envs csv> [lock_write]
job() {
  local REF="$1" LABEL="$2" INTO="$3" AS="$4" ENVS="$5" LW="${6:-0}"
  local TODO; TODO=$(missing_cells "$INTO" "$AS" "$ENVS")
  [ -z "$TODO" ] && { log "$LABEL: all of [$ENVS] present on $INTO; skip"; return 0; }
  for attempt in $(seq 1 36); do
    local RUN="$(date -u +%Y%m%dT%H%MZ)-${LABEL}-agentic"
    log "$LABEL: start $RUN envs=[$TODO] -> $INTO/$AS (attempt $attempt)"
    BENCHSUITE_LOCK_WRITE="$LW" BENCHSUITE_MERGE_INTO="$INTO" BENCHSUITE_MERGE_AS="$AS" BENCHSUITE_CHAT_ENVS="$TODO" \
      bash "$HERE/pass.sh" "$REF" "$LABEL" "$RUN" agentic > "$HERE/state/pass-$RUN.log" 2>&1
    local CODE; CODE=$(cat "$HERE/state/pass-$RUN.exit" 2>/dev/null || echo 1)
    if [ "$CODE" = "2" ]; then
      rm -f "$HERE/state/pass-$RUN.log" "$HERE/state/pass-$RUN.exit" "$HERE/state/pass-$RUN.pid"
      log "$LABEL: no Lium stock; retry in 10 min"; sleep 600; continue
    fi
    log "$LABEL: $RUN ended exit=$CODE"; return 0
  done
}

# swe_pro_job <ref> <label> <merge_into> <merge_as>: only with >= 2 spare 1x H200 listed
swe_pro_job() {
  [ -n "$(missing_cells "$3" "$4" swebench-pro)" ] || { log "$2: swebench-pro present; skip"; return 0; }
  while :; do
    local N; N=$("$PY" "$HERE/kingpod.py" stock --plan h200-1x 2>/dev/null | tail -1)
    if [ "${N:-0}" -ge 2 ]; then break; fi
    log "$2: swebench-pro waits for spare H200 stock (listed: ${N:-0}, need 2); check again in 15 min"; sleep 900
  done
  job "$1" "$2" "$3" "$4" swebench-pro
}

wait_for() { while [ ! -f "$HERE/state/pass-$1.exit" ]; do sleep 120; done; }

worker_a() {
  wait_for 20260915T1745Z-6d0ee567e33e-agentic   # the first-look pass (TB2 + tau2 airline) merges first
  job "$R13" 13      20260915T0753Z-6d0ee567e33e king    "$STD" 1     # first job re-pins the lock (tau3 side venv)
  wait_for 20260915T1815Z-genesis-agentic
  job "$GEN" genesis 20260915T1415Z-genesis     king    "$STD"
  job "$R11" 11      20260912T1455Z-0ce59769300c king    "$STD"
  # SWE-bench Pro, one model at a time, behind the H200 demand
  swe_pro_job "$R13" 13      20260915T0753Z-6d0ee567e33e king
  swe_pro_job "$TEA" teacher 20260912T1455Z-0ce59769300c teacher
}
worker_b() {
  sleep 1500   # let worker A re-pin the lock first
  job "$TEA" teacher 20260912T1455Z-0ce59769300c teacher "$STD"
  job "$U69" 12      20260914T2152Z-6c877fd2242d king    "$STD"
}

worker_a > "$HERE/state/agentic-queue-a.log" 2>&1 &
worker_b > "$HERE/state/agentic-queue-b.log" 2>&1 &
wait
