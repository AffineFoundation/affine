#!/bin/bash
# Fast pass: the whole card for one model in parallel (Jacob 2026-09-19, target <= 2.5 h
# crown -> full card). Topology (suite.toml [fast]):
#   3 chat pods   the chat suite sharded by measured wall time, then the long-context docker
#                 cells (graphwalks / oolong+mrcr) and miniF2F (Prime) behind the shards
#   agentic pod   tau2 x3 + tau3 on the pod; TB2 (Harbor -> Daytona) and Gaia2 (ARE) from the box
#   swe pod       two-replica server for SWE-bench Verified on Daytona at [fast].swe_in_flight
# Pods rent + install in parallel; every finished cell publishes (5-min poll, light pull);
# the final publish is the same operation. Cost ~ 5 pods x ~2.5 h.
#   fast_pass.sh <ref (digest | hf://repo@rev)> <label> <run_id>       (env: CHALLENGER_* pass through)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="${BENCHSUITE_PYTHON:-$REPO/.venv/bin/python}"
BENCH_HOME="${BENCH_HOME:-$HOME/benchsuite}"
REF="$1"; LABEL="$2"; RUN_ID="$3"
toml() { "$PY" -c 'import tomllib,sys,json; d=tomllib.load(open("'"$HERE"'/suite.toml","rb")); v=d
for k in sys.argv[1].split("."): v=v[k] if not k.isdigit() else v[int(k)]
print(json.dumps(v) if isinstance(v,(list,dict)) and any(isinstance(x,(list,dict)) for x in (v if isinstance(v,list) else v.values())) else (",".join(v) if isinstance(v,list) else v))' "$1"; }
log() { echo "[fast-pass] $(date -u +%FT%TZ) $*"; }
mkdir -p "$HERE/state"
LOG_EXIT="$HERE/state/pass-$RUN_ID.exit"
echo $$ > "$HERE/state/pass-$RUN_ID.pid"
finish() { echo "$1" > "$LOG_EXIT"; exit "$1"; }
export LIUM_API_KEY="${LIUM_API_KEY:-${LIUM:-}}"
export PRIME_API_KEY="${PRIME_API_KEY:-${PRIME:-}}"
export HARBOR_BIN="${HARBOR_BIN:-$BENCH_HOME/harborenv/bin/harbor}"
export ARE_BIN="${ARE_BIN:-$BENCH_HOME/areenv/bin/are-benchmark}"
CODE_COMMIT=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)
RUN_DIR="$BENCH_HOME/runs/$RUN_ID"; mkdir -p "$RUN_DIR"
T_START=$(date +%s)
TEACHER_FROM=$(toml modes.teacher_from)
# FAST_GROUPS limits the topology (vendor-settings reference rows run chat + tb2 only);
# BENCHSUITE_SETTINGS_JSON / FAST_TB2_ARGS carry the model-card settings to the cells.
FAST_GROUPS="${FAST_GROUPS:-chat,after,agentic,tb2,gaia2,swe}"
has_group() { [[ ",$FAST_GROUPS," == *",$1,"* ]]; }
SETTINGS_JSON="${BENCHSUITE_SETTINGS_JSON:-}"
N_SHARDS=$("$PY" -c 'import tomllib; print(len(tomllib.load(open("'"$HERE"'/suite.toml","rb"))["fast"]["chat_shards"]))')
SSHO=(-i "$HOME/.ssh/id_ed25519" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$HERE/state/known_hosts" -o ConnectTimeout=20 -o LogLevel=ERROR)

# ---- secrets from the vault (like run_pass.sh)
if [ -z "${DAYTONA_API_KEY:-}" ] && [ -n "${OP_SERVICE_ACCOUNT_TOKEN:-}" ] && command -v op >/dev/null 2>&1; then
  export DAYTONA_API_KEY; DAYTONA_API_KEY=$(op read --no-newline "${DAYTONA_OP_ITEM:-op://Arbos/fywmj6vtq5delybw5c7a53l2qa/notesPlain}" 2>/dev/null | grep -o 'dtn_[A-Za-z0-9_-]*' | head -1)
fi
DOCKERHUB_OP_ITEM="${DOCKERHUB_OP_ITEM:-op://Arbos/e7y3qzb2flaczam4rindajho4m}"
if [ -z "${DOCKERHUB_TOKEN:-}" ] && [ -n "${OP_SERVICE_ACCOUNT_TOKEN:-}" ] && command -v op >/dev/null 2>&1; then
  DOCKERHUB_USER=$(op read --no-newline "$DOCKERHUB_OP_ITEM/username" 2>/dev/null) || DOCKERHUB_USER=""
  DOCKERHUB_TOKEN=$(op read --no-newline "$DOCKERHUB_OP_ITEM/credential" 2>/dev/null) || DOCKERHUB_TOKEN=""
fi

# ---- ref -> digest / hf flag
DIGEST="$REF" R2FLAG=""
if [[ "$REF" == r2://* ]]; then DIGEST="${CHALLENGER_REVISION:?CHALLENGER_REVISION required for an r2:// ref}"; R2FLAG="--r2 $REF"; export AFFINE_EVAL_R2_ENDPOINT="${AFFINE_EVAL_R2_ENDPOINT:-${R2_ENDPOINT:-}}"
elif [[ "$REF" == hf://* ]]; then SPEC="${REF#hf://}"; DIGEST="hf-$(echo "${SPEC#*@}" | cut -c1-10)"; R2FLAG="--hf $SPEC"; fi

podf() { "$PY" -c 'import json,sys; m=json.load(open("'"$HERE"'/state/pods.json"))[sys.argv[1]]; print(m[sys.argv[2]])' "$1" "$2"; }

# ---- roles: chat1..N, agentic, swe
ROLES=(); for i in $(seq 1 "$N_SHARDS"); do ROLES+=("chat$i"); done
{ has_group agentic || has_group tb2 || has_group gaia2; } && ROLES+=(agentic)
has_group swe && ROLES+=(swe)
declare -A POD PLANS
for i in $(seq 1 "$N_SHARDS"); do PLANS[chat$i]=$(toml fast.chat_plans | tr "," " "); done
PLANS[agentic]=$(toml fast.agentic_plans | tr "," " "); PLANS[swe]=$(toml fast.swe_plans | tr "," " ")
POD[agentic]=""; POD[swe]=""
cleanup() { for r in "${ROLES[@]}"; do [ -n "${POD[$r]:-}" ] && { log "releasing ${POD[$r]} ($r)"; "$PY" "$HERE/kingpod.py" release "${POD[$r]}" >/dev/null 2>&1 || true; }; done; }
trap cleanup EXIT

RENT_WAIT_S="${FAST_RENT_WAIT_S:-1500}"     # a pod that does not serve in 25 min is replaced, not waited on for 60
RENT_DEADLINE_S="${FAST_RENT_DEADLINE_S:-3300}"
rent_role() {  # role -> writes state/fast-$RUN_ID-$role.pod
  # Rent, wait, and on a dead executor release + rent the next one until the deadline.
  # 2026-09-19: reign 18's pass lost 2 of 5 pods and reign 16's all 4 chat/agentic pods to
  # executors that never served in 60 min (one try each) -> no card. Lium executors fail
  # independently (~40 % that day), so one retry per role recovers most passes.
  local role="$1" pod="" t0 plan
  t0=$(date +%s)
  while [ $(( $(date +%s) - t0 )) -lt "$RENT_DEADLINE_S" ]; do
    pod=""
    for plan in ${PLANS[$role]}; do
      # shellcheck disable=SC2086
      pod=$("$PY" "$HERE/kingpod.py" rent --plan "$plan" --digest "$DIGEST" $R2FLAG 2>>"$RUN_DIR/rent-$role.log" | tail -1) && [ -n "$pod" ] && break
      pod=""
    done
    [ -n "$pod" ] || { echo "$(date -u +%FT%TZ) $role: no stock on any plan" >> "$RUN_DIR/rent-$role.log"; sleep 120; continue; }
    echo "$pod" > "$HERE/state/fast-$RUN_ID-$role.pod"
    if timeout "$RENT_WAIT_S" "$PY" "$HERE/kingpod.py" wait "$pod" >>"$RUN_DIR/rent-$role.log" 2>&1; then return 0; fi
    # not serving: the pod may still answer (slow ssh tripped `wait` before) — probe once
    if curl -s -m 15 -H "Authorization: Bearer $(podf "$pod" key 2>/dev/null)" "$(podf "$pod" base_url 2>/dev/null)/models" 2>/dev/null | grep -q '"data"'; then return 0; fi
    echo "$(date -u +%FT%TZ) $role: $pod never served in ${RENT_WAIT_S}s; releasing with a strike and renting again" >> "$RUN_DIR/rent-$role.log"
    "$PY" "$HERE/kingpod.py" release "$pod" --strike "never served in ${RENT_WAIT_S}s (fast pass $RUN_ID)" >>"$RUN_DIR/rent-$role.log" 2>&1 || true
    echo "" > "$HERE/state/fast-$RUN_ID-$role.pod"; pod=""
  done
  [ -n "$pod" ] && echo "FAILED $pod" > "$HERE/state/fast-$RUN_ID-$role.pod" && return 3
  echo "" > "$HERE/state/fast-$RUN_ID-$role.pod"; return 2
}
log "pass $RUN_ID (fast) ref=$REF label=$LABEL code=$CODE_COMMIT: renting ${#ROLES[@]} pods in parallel"
for r in "${ROLES[@]}"; do rent_role "$r" & done
wait
for r in "${ROLES[@]}"; do
  v=$(cat "$HERE/state/fast-$RUN_ID-$r.pod" 2>/dev/null || echo "")
  case "$v" in
    FAILED*) pod="${v#FAILED }"
             # `wait` gave up, but the pod may serve anyway (a slow ssh tripped it on 2026-09-19): probe once
             if curl -s -m 15 -H "Authorization: Bearer $(podf "$pod" key)" "$(podf "$pod" base_url)/models" 2>/dev/null | grep -q '"data"'; then POD[$r]="$pod"; log "$r: wait failed but $pod answers; using it"
             else log "$r: pod never served ($pod); releasing"; "$PY" "$HERE/kingpod.py" release "$pod" >/dev/null 2>&1; fi;;
    "") log "$r: no stock";;
    *) POD[$r]="$v";;
  esac
done
[ -n "${POD[chat1]:-}" ] || { log "no chat pod at all; giving up"; finish 2; }
T_READY=$(date +%s); log "pods ready after $(( (T_READY - T_START) / 60 )) min: $(for r in "${ROLES[@]}"; do echo -n "$r=${POD[$r]:-none} "; done)"

ssh_pod() { local pod="$1"; shift; ssh "${SSHO[@]}" -p "$(podf "$pod" ssh_port)" "root@$(podf "$pod" ssh_host)" "$@"; }
scp_pod() { local pod="$1" src="$2" dst="$3"; scp "${SSHO[@]}" -P "$(podf "$pod" ssh_port)" "$src" "root@$(podf "$pod" ssh_host):$dst"; }

# ---- install the eval env on the chat + agentic pods, in parallel (the swe pod only serves)
tar -C "$REPO" -czf "/tmp/benchsuite-$RUN_ID.tgz" ops/benchsuite
install_pod() {
  local pod="$1"
  scp_pod "$pod" "/tmp/benchsuite-$RUN_ID.tgz" /tmp/benchsuite.tgz || return 4
  ssh_pod "$pod" "mkdir -p /root/affine /root/benchsuite/runs && cd /root/affine && tar xzf /tmp/benchsuite.tgz && BENCH_HOME=/root/benchsuite bash /root/affine/ops/benchsuite/install_eval_env.sh > /root/install.log 2>&1; tail -1 /root/install.log" || return 5
  [ -n "${DOCKERHUB_TOKEN:-}" ] && printf '%s' "$DOCKERHUB_TOKEN" | ssh_pod "$pod" "docker login -u '$DOCKERHUB_USER' --password-stdin >/dev/null 2>&1" || true
  ssh_pod "$pod" "cd /root/affine/ops/benchsuite && export PATH=\$HOME/.local/bin:\$PATH && /root/benchsuite/verifiers/.venv/bin/python lock.py check --bench-home /root/benchsuite" || { log "LOCK MISMATCH on $pod"; return 10; }
}
for r in "${ROLES[@]}"; do [ "$r" = swe ] && continue; [ -n "${POD[$r]:-}" ] && { install_pod "${POD[$r]}" > "$RUN_DIR/install-$r.log" 2>&1 & }; done
wait
rm -f "/tmp/benchsuite-$RUN_ID.tgz"
for r in "${ROLES[@]}"; do [ "$r" = swe ] && continue; [ -n "${POD[$r]:-}" ] || continue
  grep -q "INSTALL_DONE" "$RUN_DIR/install-$r.log" || { log "$r: install failed (see install-$r.log); dropping the pod"; "$PY" "$HERE/kingpod.py" release "${POD[$r]}" >/dev/null 2>&1; POD[$r]=""; }
done
T_INST=$(date +%s); log "installs done after $(( (T_INST - T_START) / 60 )) min"

# ---- manifest + teacher baseline on the box (one canonical manifest.json; pods write manifest-<role>.json)
"$PY" - "$REF" "$LABEL" "$CODE_COMMIT" "$TEACHER_FROM" "$RUN_DIR/manifest.json" "$(for r in "${ROLES[@]}"; do echo -n "$r=${POD[$r]:-none},"; done)" <<'PY'
import json, os, sys, time
ref, label, commit, tfrom, out, pods = sys.argv[1:]
if ref.startswith("r2://"): king = {"repo": ref, "digest": os.environ.get("CHALLENGER_REVISION", "")}
elif ref.startswith("hf://"):
    spec = ref[5:]; king = {"repo": ref, "hf_repo": spec.split("@")[0], "hf_revision": spec.partition("@")[2], "digest": "hf-" + spec.partition("@")[2][:10]}
else: king = {"digest": ref}
if label.isdigit(): king["reign"] = int(label)
else: king["label"] = label
if os.environ.get("BENCHSUITE_SETTINGS_JSON"):
    king["settings"] = json.loads(os.environ["BENCHSUITE_SETTINGS_JSON"]); king["settings_note"] = os.environ.get("BENCHSUITE_SETTINGS_NOTE", "")
if os.environ.get("BENCHSUITE_REFERENCE_ROW"): king["reference"] = os.environ["BENCHSUITE_REFERENCE_ROW"]
if os.environ.get("BENCHSUITE_DIGEST_SUFFIX"):
    # a settings row must not attach to the model's own kingboard row (matched by digest /
    # hf_revision): suffix the digest and keep the revision under another key
    king["digest"] = king.get("digest", "") + os.environ["BENCHSUITE_DIGEST_SUFFIX"]
    if "hf_revision" in king: king["base_hf_revision"] = king.pop("hf_revision")
duel = {k: os.environ.get(f"CHALLENGER_{k.upper()}") for k in ("margin", "z", "vs_reign", "vs_king_digest", "judged_at", "hotkey")}
if any(duel.values()): king["duel"] = {k: (float(v) if k in ("margin", "z") and v not in (None, "") else v) for k, v in duel.items()}
m = {"run_id": os.path.basename(os.path.dirname(out)), "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
     "mode": "fast", "king": king, "teacher": {"reused_from": tfrom},
     "where": {"provider": "Lium (our fleet, TAO) + Daytona", "topology": "fast: 3 chat shards + agentic pod + 2-replica swe server; SWE/TB2 on Daytona, Gaia2 via ARE", "pods": pods},
     "code": {"affine_commit": commit}, "cells": {}}
json.dump(m, open(out, "w"), indent=1)
PY
[ -d "$BENCH_HOME/runs/$TEACHER_FROM/teacher" ] && rsync -a --exclude 'traces.jsonl*' --exclude 'logs' --exclude 'eval.log' --exclude 'harbor' --exclude 'are' "$BENCH_HOME/runs/$TEACHER_FROM/teacher/" "$RUN_DIR/teacher/"

pod_env() {  # role -> exports for a pod-side run
  local pod="${POD[$1]}"; echo "export BENCH_API_KEY='$(podf "$pod" key)' PRIME_API_KEY='${PRIME_API_KEY:-}' HF_TOKEN='${HF_TOKEN:-}' BENCHSUITE_CHAT_IMAGE=affine-bench-chat:py311 BENCHSUITE_SETTINGS_JSON='$SETTINGS_JSON'; cd /root/affine/ops/benchsuite"
}
PYR=/root/benchsuite/verifiers/.venv/bin/python
suite_on_pod() {  # role envs runtime temps concurrency manifest parallel
  local pod="${POD[$1]}" url="http://127.0.0.1:$(podf "${POD[$1]}" front_internal)/v1" served="$(podf "${POD[$1]}" served)"
  ssh_pod "$pod" "$(pod_env "$1") && $PYR run_suite.py run --run-id $RUN_ID --key-env BENCH_API_KEY --king-url $url --king-model $served --models king --teacher-from $TEACHER_FROM --verifiers-dir /root/benchsuite/verifiers --out /root/benchsuite/runs --runtime $3 --envs $2 --temps $4 --concurrency $5 --parallel-envs $7 --manifest $6 --push"
}
pull_light_all() { for r in "${ROLES[@]}"; do [ "$r" = swe ] && continue; [ -n "${POD[$r]:-}" ] || continue
  ssh_pod "${POD[$r]}" "cd /root/benchsuite/runs && tar czf - --exclude='*/logs' --exclude='*/traces.jsonl*' --exclude='*/eval.log' --exclude='*/harbor' --exclude='*/are' --exclude='manifest.json' $RUN_ID 2>/dev/null" 2>/dev/null | tar xzf - -C "$BENCH_HOME/runs" 2>/dev/null; done; }
pull_all() { for r in "${ROLES[@]}"; do [ "$r" = swe ] && continue; [ -n "${POD[$r]:-}" ] || continue
  ssh_pod "${POD[$r]}" "cd /root/benchsuite/runs && tar czf - --exclude='*/logs/attempt_*' --exclude='manifest.json' $RUN_ID 2>/dev/null" 2>/dev/null | tar xzf - -C "$BENCH_HOME/runs" 2>/dev/null; done; }
publish_partial() { "$PY" "$HERE/publish.py" --run-dir "$RUN_DIR" --only-state --partial --eta >/dev/null 2>&1 || true; }

# ---- launch every group at once
PIDS=()
for i in $(seq 1 "$N_SHARDS"); do
  r="chat$i"; [ -n "${POD[$r]:-}" ] || { log "$r: no pod; its shard is skipped"; continue; }
  SHARD=$(toml "fast.chat_shards.$((i-1))" | tr -d '[]" ' ); AFTER=$(toml "fast.shard_after.$((i-1))" | tr -d '[]" ')
  ( suite_on_pod "$r" "$SHARD" docker primary,secondary 64 "manifest-$r.json" 2
    has_group after || AFTER=""
    for e in ${AFTER//,/ }; do
      rt=docker; [ "$e" = minif2f ] && rt=prime
      suite_on_pod "$r" "$e" "$rt" primary 48 "manifest-$r-after.json" 1
    done ) > "$RUN_DIR/$r.log" 2>&1 &
  PIDS+=($!)
done
if [ -n "${POD[agentic]:-}" ]; then
  if has_group agentic; then
    ( suite_on_pod agentic "$(toml fast.agentic_envs_on_pod)" docker primary 8 manifest-agentic.json 4 ) > "$RUN_DIR/agentic.log" 2>&1 &
    PIDS+=($!)
  fi
  AURL="$(podf "${POD[agentic]}" base_url)"; ASERVED="$(podf "${POD[agentic]}" served)"
  # TB2 (Daytona) and Gaia2 (ARE) run CONCURRENTLY against the agentic pod (the first run
  # chained them and Gaia2 only started after TB2's long tail)
  if has_group tb2; then
    # shellcheck disable=SC2086
    ( export BENCH_API_KEY="$(podf "${POD[agentic]}" key)"
      [ -n "${DAYTONA_API_KEY:-}" ] && "$PY" "$HERE/harbor_cell.py" run --env terminal-bench-2 --model "$ASERVED" --model-label king --model-url "$AURL" --model-key-env BENCH_API_KEY --out "$RUN_DIR/king" --concurrency "${FAST_TB2_IN_FLIGHT:-$(toml fast.tb2_in_flight)}" --agent-timeout-s "${FAST_TB2_TIMEOUT_S:-3600}" ${FAST_TB2_ARGS:-} ) > "$RUN_DIR/agentic-box.log" 2>&1 &
    PIDS+=($!)
  fi
  if has_group gaia2; then
    ( export BENCH_API_KEY="$(podf "${POD[agentic]}" key)"
      [ -n "${PRIME_API_KEY:-}" ] && "$PY" "$HERE/gaia2_cell.py" run --model "$ASERVED" --model-label king --model-url "$AURL" --model-key-env BENCH_API_KEY --judge-key-env PRIME_API_KEY --concurrency "$(toml fast.gaia2_in_flight)" --out "$RUN_DIR/king" ) > "$RUN_DIR/gaia2-box.log" 2>&1 &
    PIDS+=($!)
  fi
fi
if [ -n "${POD[swe]:-}" ] && [ -n "${DAYTONA_API_KEY:-}" ]; then
  # in flight = 64 per served replica (reign 18: 200 agents on ONE B200 replica spent the
  # 1-h budget queueing -> 454/500 timeouts); [fast].swe_in_flight is the cap for a 2x+ pod
  SWE_REPLICAS=$("$PY" -c 'import json; m=json.load(open("'"$HERE"'/state/pods.json"))["'"${POD[swe]}"'"]; print(int((m.get("plan") or {}).get("replicas") or 1))')
  SWE_INFLIGHT=$(( 64 * SWE_REPLICAS )); [ "$SWE_INFLIGHT" -gt "$(toml fast.swe_in_flight)" ] && SWE_INFLIGHT=$(toml fast.swe_in_flight)
  ( export BENCH_API_KEY="$(podf "${POD[swe]}" key)"
    "$PY" "$HERE/harbor_cell.py" run --env swebench-verified --model "$(podf "${POD[swe]}" served)" --model-label king --model-url "$(podf "${POD[swe]}" base_url)" --model-key-env BENCH_API_KEY --out "$RUN_DIR/king" --concurrency "$SWE_INFLIGHT" --agent-timeout-s "$(toml sandbox_daytona.budgets.default.agent_timeout_s)" ) > "$RUN_DIR/swe-box.log" 2>&1 &
  PIDS+=($!)
fi
log "launched ${#PIDS[@]} groups; polling every $(toml fast.poll_s) s"

# ---- poll: light pull + partial publish until every group ends
while :; do
  alive=0; for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && alive=$((alive+1)); done
  pull_light_all; publish_partial
  [ "$alive" -eq 0 ] && break
  sleep "$(toml fast.poll_s)"
done
wait
T_CELLS=$(date +%s); log "all groups ended after $(( (T_CELLS - T_START) / 60 )) min"

# ---- tail: pull everything, retry infra errors per pod, summarize, publish, release
pull_all
for r in "${ROLES[@]}"; do [ "$r" = swe ] && continue; [ -n "${POD[$r]:-}" ] || continue
  ssh_pod "${POD[$r]}" "$(pod_env "$r") && $PYR run_suite.py retry --run-id $RUN_ID --out /root/benchsuite/runs --verifiers-dir /root/benchsuite/verifiers" > "$RUN_DIR/retry-$r.log" 2>&1 &
done
wait; pull_all
"$PY" "$HERE/run_suite.py" summarize --run-id "$RUN_ID" --out "$BENCH_HOME/runs" >/dev/null 2>&1 || true
"$PY" - "$RUN_DIR/manifest.json" "$T_START" "$T_READY" "$T_INST" "$T_CELLS" "$(date +%s)" <<'PY'
import json, sys
p, *ts = sys.argv[1:]; t0, t_ready, t_inst, t_cells, t_end = map(int, ts)
m = json.load(open(p)); m["timing"] = {"pods_ready_min": (t_ready - t0) // 60, "installs_done_min": (t_inst - t0) // 60,
                                        "cells_done_min": (t_cells - t0) // 60, "total_min": (t_end - t0) // 60}
json.dump(m, open(p, "w"), indent=1)
PY
"$PY" "$HERE/publish.py" --run-dir "$RUN_DIR" || finish 8
log "card complete: $(( ($(date +%s) - T_START) / 60 )) min crown-pass-start -> full card"
finish 0
