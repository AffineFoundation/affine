#!/bin/bash
# Duel-faithful bench against the swarm router (or any base URL).
# Usage: ./bench_swarm.sh [base_url] [concurrency] [n_turns] [out_name]
set -euo pipefail
BASE=${1:-http://127.0.0.1:9100/v1}
CONC=${2:-24}
TURNS=${3:-48}
OUT=${4:-bench_swarm_latest}
cd /home/const/subnet120
source .venv/bin/activate
python ops/teacher-b200/bench_duel_real.py \
  --base-url "$BASE" \
  --concurrency "$CONC" \
  --n-turns "$TURNS" \
  --out "ops/teacher-swarm/state/${OUT}.json"
