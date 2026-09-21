#!/bin/bash
# After reign 13's 4 h / 250-call SWE-bench cell lands, run the same cell for reign 14
# and the genesis (Jacob 2026-09-17 14:43 UTC) so the Albedo comparison row (56.2 @4h)
# has all four of ours next to it. Each model: a two-replica Lium pod (h200-2x / b200-2x
# by stock, then the single-replica plans) serving only; 100 tasks in flight on Daytona;
# the cell merges into the model's existing card as swebench-verified@4h250.
#   bash daytona_4h_queue.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="${BENCHSUITE_PYTHON:-$REPO/.venv/bin/python}"
BENCH_HOME="${BENCH_HOME:-$HOME/benchsuite}"
# shellcheck disable=SC1091
source "$HERE/env.sh"
export LIUM_API_KEY="${LIUM_API_KEY:-${LIUM:-}}"
export HARBOR_BIN="${HARBOR_BIN:-$BENCH_HOME/harborenv/bin/harbor}"
log() { echo "[daytona-4h] $(date -u +%FT%TZ) $*"; }
toml() { "$PY" -c 'import tomllib,sys; d=tomllib.load(open("'"$HERE"'/suite.toml","rb")); v=d
for k in sys.argv[1].split("."): v=v[k]
print(",".join(v) if isinstance(v,list) else v)' "$1"; }
if [ -z "${DAYTONA_API_KEY:-}" ]; then
  export DAYTONA_API_KEY; DAYTONA_API_KEY=$(op read --no-newline "${DAYTONA_OP_ITEM:-op://Arbos/fywmj6vtq5delybw5c7a53l2qa/notesPlain}" 2>/dev/null | grep -o 'dtn_[A-Za-z0-9_-]*' | head -1)
fi
[ -n "${DAYTONA_API_KEY:-}" ] || { log "no DAYTONA_API_KEY"; exit 1; }
TAG=4h250
TIMEOUT_S=$(toml sandbox_daytona.budgets.4h250.agent_timeout_s)
STEPS=$(toml sandbox_daytona.budgets.4h250.step_limit)
CONC=$(toml sandbox_daytona.concurrency)
PLANS="$(toml sandbox_daytona.pod_plans | tr "," " ") b200-1x h200-1x pro6000-1x"

# wait for reign 13's cell (the validation run) to be final
R13_CELL="$BENCH_HOME/runs/20260915T0753Z-6d0ee567e33e/king/swebench-verified@${TAG}__t0/summary.json"
until [ -f "$R13_CELL" ] && [ "$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["n"])' "$R13_CELL")" -ge 490 ]; do
  log "waiting for reign 13's @$TAG cell"; sleep 900
done

latest_run_dir() {  # <digest12 or label> -> newest non-agentic run dir under runs/
  ls -td "$BENCH_HOME/runs/"*"-$1" "$BENCH_HOME/runs/"*"-$1-"[0-9]* 2>/dev/null | grep -v -- "-agentic\|-challenger\|-w2c" | head -1
}

one() {  # <ref (digest|hf://)> <label> <run_dir> [hf flag]
  local REF="$1" LABEL="$2" INTO="$3" HFFLAG="${4:-}" POD=""
  [ -d "$INTO" ] || { log "$LABEL: no card run dir ($INTO); skip"; return 1; }
  [ -f "$INTO/king/swebench-verified@${TAG}__t0/summary.json" ] && { log "$LABEL: @$TAG cell present; skip"; return 0; }
  local DIGEST="$REF"
  if [[ "$REF" == hf://* ]]; then local SPEC="${REF#hf://}"; DIGEST="hf-$(echo "${SPEC#*@}" | cut -c1-10)"; HFFLAG="--hf $SPEC"; fi
  for PLAN in $PLANS; do
    # shellcheck disable=SC2086
    POD=$("$PY" "$HERE/kingpod.py" rent --plan "$PLAN" --digest "$DIGEST" $HFFLAG 2>/dev/null | tail -1) && [ -n "$POD" ] && break
    POD=""
  done
  [ -n "$POD" ] || { log "$LABEL: no Lium stock on any plan; retry later"; return 2; }
  "$PY" "$HERE/kingpod.py" wait "$POD" > /dev/null || { log "$LABEL: pod never served"; "$PY" "$HERE/kingpod.py" release "$POD"; return 3; }
  local URL KEY SERVED
  URL=$("$PY" -c 'import json; print(json.load(open("'"$HERE"'/state/pods.json"))["'"$POD"'"]["base_url"])')
  KEY=$("$PY" -c 'import json; print(json.load(open("'"$HERE"'/state/pods.json"))["'"$POD"'"]["key"])')
  SERVED=$("$PY" -c 'import json; print(json.load(open("'"$HERE"'/state/pods.json"))["'"$POD"'"]["served"])')
  export BENCH_API_KEY="$KEY"
  log "$LABEL: $POD ($URL) serving; SWE-bench Verified @$TAG on Daytona, $CONC in flight -> $INTO"
  "$PY" "$HERE/harbor_cell.py" run --env swebench-verified --budget-tag "$TAG" --agent-timeout-s "$TIMEOUT_S" --step-limit "$STEPS" \
      --concurrency "$CONC" --model "$SERVED" --model-label king --model-url "$URL" --model-key-env BENCH_API_KEY --out "$INTO/king"
  "$PY" - "$INTO/king/swebench-verified@${TAG}__t0/summary.json" "$POD" "$HERE/state/pods.json" <<'PY'
import json, sys
p, pod, pods_path = sys.argv[1:]; s = json.load(open(p)); m = json.load(open(pods_path)).get(pod, {})
s["where"] = {"provider": "Lium (our fleet, TAO)", "pod_id": pod, "gpu": m.get("machine"), "plan": (m.get("plan") or {}).get("name"),
              "usd_per_hour": m.get("price"), "note": "pod serves the model only; task containers on Daytona"}
json.dump(s, open(p, "w"), indent=1)
PY
  "$PY" "$HERE/publish.py" --run-dir "$INTO" --only-cells "king/swebench-verified@${TAG}__t0" || log "$LABEL: publish failed"
  log "$LABEL: releasing $POD"; "$PY" "$HERE/kingpod.py" release "$POD" || true
}

R14=0f4029fd59ededd2d312a18080ea8b3aa941f3a48c32
R14_FULL=$("$PY" -c 'import json; print(json.load(open("'"$REPO"'/affine/state/state.json"))["king"]["revision"])')
[ "${R14_FULL:0:12}" = "0f4029fd59ed" ] || log "note: state.json king is ${R14_FULL:0:12}, not reign 14; using the recorded reign-14 digest"
R14_REF="${R14_FULL}"; [ "${R14_FULL:0:12}" = "0f4029fd59ed" ] || R14_REF="$R14"
for attempt in $(seq 1 12); do
  one "$R14_REF" 14 "$(latest_run_dir 0f4029fd59ed)" && break
  sleep 900
done
for attempt in $(seq 1 12); do
  one "hf://Qwen/Qwen3.6-35B-A3B@995ad96eacd98c81ed38be0c5b274b04031597b0" genesis "$BENCH_HOME/runs/20260915T1415Z-genesis" && break
  sleep 900
done
log "done"
