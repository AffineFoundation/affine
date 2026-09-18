#!/bin/bash
# Ask-hole cells (modes.ask_envs = gaia2-ambiguity) for the comparison set — teacher,
# genesis, reign 13, reign 14 — merged into their cards (Jacob 2026-09-18, cap $120).
# Teacher via Engy (hosted; no pod); the others on a rented Lium pod that only serves.
# Coordinated with the coverage queue by construction: kingpod pods are per-model, no SWE,
# no docker on the pod.
#   bash ask_queue.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="${BENCHSUITE_PYTHON:-$REPO/.venv/bin/python}"
BENCH_HOME="${BENCH_HOME:-$HOME/benchsuite}"
# shellcheck disable=SC1091
source "$HERE/env.sh"
export LIUM_API_KEY="${LIUM_API_KEY:-${LIUM:-}}"
export PRIME_API_KEY="${PRIME_API_KEY:-${PRIME:-}}"
export ARE_BIN="${ARE_BIN:-$BENCH_HOME/areenv/bin/are-benchmark}"
log() { echo "[ask-queue] $(date -u +%FT%TZ) $*"; }
toml() { "$PY" -c 'import tomllib,sys; d=tomllib.load(open("'"$HERE"'/suite.toml","rb")); v=d
for k in sys.argv[1].split("."): v=v[k]
print(",".join(v) if isinstance(v,list) else v)' "$1"; }
CONC=16
TEACHER_RUN=20260912T1455Z-0ce59769300c

latest_run_dir() { ls -td "$BENCH_HOME/runs/"*"-$1" "$BENCH_HOME/runs/"*"-$1-"[0-9]* 2>/dev/null | grep -v -- "-agentic\|-challenger\|-w2c" | head -1; }

teacher() {
  local INTO="$BENCH_HOME/runs/$TEACHER_RUN"
  [ -f "$INTO/teacher/gaia2-ambiguity__t0/summary.json" ] && { log "teacher: present; skip"; return 0; }
  log "teacher: Gaia2 ambiguity via Engy"
  "$PY" "$HERE/gaia2_cell.py" run --model qwen3.8-27b --model-label teacher --model-url https://api.engy.ai/v1 --model-key-env ENGY \
      --judge-key-env PRIME_API_KEY --concurrency "$CONC" --out "$INTO/teacher" || return 1
  "$PY" - "$INTO/teacher/gaia2-ambiguity__t0/summary.json" <<'PY'
import json, sys
p = sys.argv[1]; s = json.load(open(p))
s["served_by"] = {"provider": "Engy (hosted)", "base_url": "https://api.engy.ai/v1", "model": "qwen3.8-27b"}
json.dump(s, open(p, "w"), indent=1)
PY
  "$PY" "$HERE/publish.py" --run-dir "$INTO" --only-cells "teacher/gaia2-ambiguity__t0"
}

king() {  # <ref (digest | hf://repo@rev)> <label> <run_dir>
  local REF="$1" LABEL="$2" INTO="$3" POD="" HFFLAG="" DIGEST="$1"
  [ -d "$INTO" ] || { log "$LABEL: no card run dir; skip"; return 1; }
  [ -f "$INTO/king/gaia2-ambiguity__t0/summary.json" ] && { log "$LABEL: present; skip"; return 0; }
  if [[ "$REF" == hf://* ]]; then local SPEC="${REF#hf://}"; DIGEST="hf-$(echo "${SPEC#*@}" | cut -c1-10)"; HFFLAG="--hf $SPEC"; fi
  for PLAN in $(toml modes.lium_plan) $(toml modes.lium_plan_fallbacks | tr "," " "); do
    # shellcheck disable=SC2086
    POD=$("$PY" "$HERE/kingpod.py" rent --plan "$PLAN" --digest "$DIGEST" $HFFLAG 2>/dev/null | tail -1) && [ -n "$POD" ] && break
    POD=""
  done
  [ -n "$POD" ] || { log "$LABEL: no Lium stock; retry later"; return 2; }
  "$PY" "$HERE/kingpod.py" wait "$POD" > /dev/null || { log "$LABEL: pod never served"; "$PY" "$HERE/kingpod.py" release "$POD"; return 3; }
  local URL KEY SERVED
  URL=$("$PY" -c 'import json; print(json.load(open("'"$HERE"'/state/pods.json"))["'"$POD"'"]["base_url"])')
  KEY=$("$PY" -c 'import json; print(json.load(open("'"$HERE"'/state/pods.json"))["'"$POD"'"]["key"])')
  SERVED=$("$PY" -c 'import json; print(json.load(open("'"$HERE"'/state/pods.json"))["'"$POD"'"]["served"])')
  export BENCH_API_KEY="$KEY"
  log "$LABEL: $POD serving; Gaia2 ambiguity -> $INTO"
  "$PY" "$HERE/gaia2_cell.py" run --model "$SERVED" --model-label king --model-url "$URL" --model-key-env BENCH_API_KEY \
      --judge-key-env PRIME_API_KEY --concurrency "$CONC" --out "$INTO/king"
  local RC=$?
  "$PY" "$HERE/publish.py" --run-dir "$INTO" --only-cells "king/gaia2-ambiguity__t0" || log "$LABEL: publish failed"
  log "$LABEL: releasing $POD"; "$PY" "$HERE/kingpod.py" release "$POD" || true
  return $RC
}

teacher > "$HERE/state/ask-teacher.log" 2>&1 &
R13=6d0ee567e33ee44fc238cc4a1eaae3ad9c98e3e63669548bc2fa75b94c5ebb47
R14=$("$PY" -c 'import json; k=json.load(open("'"$REPO"'/affine/state/state.json"))["king"]; print(k["revision"] if k["revision"].startswith("0f4029fd59ed") else "")')
GEN="hf://Qwen/Qwen3.6-35B-A3B@995ad96eacd98c81ed38be0c5b274b04031597b0"
for attempt in $(seq 1 12); do king "$R13" 13 "$BENCH_HOME/runs/20260915T0753Z-6d0ee567e33e" && break; sleep 900; done
if [ -n "$R14" ]; then for attempt in $(seq 1 12); do king "$R14" 14 "$(latest_run_dir 0f4029fd59ed)" && break; sleep 900; done; else log "reign 14 digest not resolvable from state.json; skipped"; fi
for attempt in $(seq 1 12); do king "$GEN" genesis "$BENCH_HOME/runs/20260915T1415Z-genesis" && break; sleep 900; done
wait
log "done"
