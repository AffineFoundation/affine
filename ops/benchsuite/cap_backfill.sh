#!/bin/bash
# Re-run the cap-bound chat cells of an EXISTING card at the new completion caps
# (suite.toml 2026-09-19: MMLU-Pro / GPQA / MATH-500 32k, LiveCodeBench 64k) on one
# rented Lium pod, keeping the old-cap cells as <env>@<cap>k (recap.py). The new cells
# land in the same run dir, so the card keeps its run_id and fills in per cell.
#
#   cap_backfill.sh <ref> <label> <run_id> [king|teacher]
#     ref     public digest | hf://repo@rev (genesis, teacher) | r2://... (CHALLENGER_REVISION set)
#     label   reign number or card label (teacher: "teacher")
#     run_id  the card's run id under $BENCH_HOME/runs (teacher: modes.teacher_from)
#   env: CAP_ENVS (default mmlu-pro,gpqa-diamond,math500,livecodebench) CAP_PLANS CAP_PARALLEL
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="${BENCHSUITE_PYTHON:-$REPO/.venv/bin/python}"
BENCH_HOME="${BENCH_HOME:-$HOME/benchsuite}"
REF="$1"; LABEL="$2"; RUN_ID="$3"; SIDE="${4:-king}"
CAP_ENVS="${CAP_ENVS:-mmlu-pro,gpqa-diamond,math500,livecodebench}"
CAP_PLANS="${CAP_PLANS:-pro6000-1x h200-1x b200-1x prime-h200-1x prime-h100-2x prime-a100-2x}"   # prime-* = Prime pods (chat cells) when Lium is out
CAP_PARALLEL="${CAP_PARALLEL:-2}"
toml() { "$PY" -c 'import tomllib,sys; d=tomllib.load(open("'"$HERE"'/suite.toml","rb")); v=d
for k in sys.argv[1].split("."): v=v[k]
print(",".join(v) if isinstance(v,list) else v)' "$1"; }
log() { echo "[cap-backfill] $(date -u +%FT%TZ) $*"; }
mkdir -p "$HERE/state"
TAG="capfill-$RUN_ID-$SIDE"
echo $$ > "$HERE/state/pass-$TAG.pid"
finish() { echo "$1" > "$HERE/state/pass-$TAG.exit"; exit "$1"; }
export LIUM_API_KEY="${LIUM_API_KEY:-${LIUM:-}}"
export PRIME_API_KEY="${PRIME_API_KEY:-${PRIME:-}}"
RUN_DIR="$BENCH_HOME/runs/$RUN_ID"
if [ ! -d "$RUN_DIR" ]; then
  # CAP_NEW_RUN=1: a fresh card for this model (old-cap variant cells live in their own run id so the
  # pull never overwrites the current-cap cells; the board merges cards by digest)
  [ "${CAP_NEW_RUN:-0}" = 1 ] || { log "no run dir $RUN_DIR"; finish 2; }
  mkdir -p "$RUN_DIR/king"
  "$PY" - "$REF" "$LABEL" "$RUN_DIR/manifest.json" "${CAP_RUN_NOTE:-}" <<'PY'
import json, os, sys, time
ref, label, out, note = sys.argv[1:]
if ref.startswith("hf://"):
    spec = ref[5:]; king = {"repo": ref, "hf_repo": spec.split("@")[0], "hf_revision": spec.partition("@")[2], "digest": "hf-" + spec.partition("@")[2][:10]}
elif ref.startswith("r2://"): king = {"repo": ref, "digest": os.environ.get("CHALLENGER_REVISION", "")}
else: king = {"digest": ref}
if label.isdigit(): king["reign"] = int(label)
else: king["label"] = label
json.dump({"run_id": os.path.basename(os.path.dirname(out)), "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "mode": "capfill",
           "king": king, "teacher": {}, "where": {"provider": "Lium (our fleet, TAO)", "topology": "one pod, cap_backfill.sh"}, "note": note, "cells": {}}, open(out, "w"), indent=1)
PY
fi
TEACHER_FROM=$(toml modes.teacher_from)
SSHO=(-i "$HOME/.ssh/id_ed25519" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$HERE/state/known_hosts" -o ConnectTimeout=20 -o LogLevel=ERROR)
DOCKERHUB_OP_ITEM="${DOCKERHUB_OP_ITEM:-op://Arbos/e7y3qzb2flaczam4rindajho4m}"
if [ -z "${DOCKERHUB_TOKEN:-}" ] && [ -n "${OP_SERVICE_ACCOUNT_TOKEN:-}" ] && command -v op >/dev/null 2>&1; then
  DOCKERHUB_USER=$(op read --no-newline "$DOCKERHUB_OP_ITEM/username" 2>/dev/null) || DOCKERHUB_USER=""
  DOCKERHUB_TOKEN=$(op read --no-newline "$DOCKERHUB_OP_ITEM/credential" 2>/dev/null) || DOCKERHUB_TOKEN=""
fi

DIGEST="$REF" R2FLAG=""
if [[ "$REF" == r2://* ]]; then DIGEST="${CHALLENGER_REVISION:?}"; R2FLAG="--r2 $REF"; export AFFINE_EVAL_R2_ENDPOINT="${AFFINE_EVAL_R2_ENDPOINT:-${R2_ENDPOINT:-}}"
elif [[ "$REF" == hf://* ]]; then SPEC="${REF#hf://}"; DIGEST="hf-$(echo "${SPEC#*@}" | cut -c1-10)"; R2FLAG="--hf $SPEC"; fi
podf() { "$PY" -c 'import json,sys; m=json.load(open("'"$HERE"'/state/pods.json"))[sys.argv[1]]; print(m.get(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else ""))' "$1" "$2" "${3:-}"; }
tool_for_plan() { case "$1" in prime-*) echo "$HERE/primepod.py";; *) echo "$HERE/kingpod.py";; esac; }
tool_for_pod() { [ "$(podf "$1" provider lium)" = prime ] && echo "$HERE/primepod.py" || echo "$HERE/kingpod.py"; }
pod_ssh_key() { [ "$(podf "$1" provider lium)" = prime ] && echo "${PRIME_SSH_KEY:-$HOME/.ssh/prime_bench}" || echo "$HOME/.ssh/id_ed25519"; }

# ---- 1. old-cap cells -> <env>@<cap>k (the plain names are free for the new cells)
"$PY" "$HERE/recap.py" --run-dir "$RUN_DIR"
"$PY" "$HERE/publish.py" --run-dir "$RUN_DIR" --only-state --partial >/dev/null 2>&1 || true

# ---- 2. one pod, replaced when an executor never serves
POD=""
cleanup() { [ -n "$POD" ] && { log "releasing $POD"; "$PY" "$(tool_for_pod "$POD")" release "$POD" >/dev/null 2>&1 || true; }; }
trap cleanup EXIT
T0=$(date +%s)
while [ $(( $(date +%s) - T0 )) -lt "${CAP_RENT_DEADLINE_S:-14400}" ]; do   # zero Lium stock is normal while a fast pass holds 5 pods: wait up to 4 h
  for plan in $CAP_PLANS; do
    # shellcheck disable=SC2086
    POD=$("$PY" "$(tool_for_plan "$plan")" rent --plan "$plan" --digest "$DIGEST" $R2FLAG 2>>"$RUN_DIR/capfill-rent.log" | tail -1) && [ -n "$POD" ] && break
    POD=""
  done
  [ -n "$POD" ] || { log "no stock; retry in 3 min"; sleep 180; continue; }
  if timeout 1500 "$PY" "$(tool_for_pod "$POD")" wait "$POD" >>"$RUN_DIR/capfill-rent.log" 2>&1; then break; fi
  if [ "$(podf "$POD" provider lium)" != prime ] && curl -s -m 15 -H "Authorization: Bearer $(podf "$POD" key)" "$(podf "$POD" base_url)/models" 2>/dev/null | grep -q '"data"'; then break; fi
  log "$POD never served in 25 min; strike + re-rent"
  "$PY" "$(tool_for_pod "$POD")" release "$POD" --strike "never served in 1500s (cap backfill $RUN_ID)" >/dev/null 2>&1 || true
  POD=""
done
[ -n "$POD" ] || { log "no serving pod within the rent deadline; giving up"; finish 3; }
log "pod $POD serves $(podf "$POD" served) after $(( ($(date +%s) - T0) / 60 )) min"
ssh_pod() { local u; u="$(podf "$POD" ssh_user root)"; local cmd="$*"; [ "$u" != root ] && cmd="sudo -n bash -c $(printf '%q' "$cmd")"   # Prime images: sudo user, layout under /root
  ssh "${SSHO[@]}" -i "$(pod_ssh_key "$POD")" -p "$(podf "$POD" ssh_port)" "$u@$(podf "$POD" ssh_host)" "$cmd"; }
RUNTIME="${CAP_RUNTIME:-docker}"; [ "$(podf "$POD" provider lium)" = prime ] && RUNTIME=prime   # CAP_RUNTIME=prime for miniF2F (Lean image only on Prime sandboxes); no docker on Prime images

# ---- 3. install + lock check
tar -C "$REPO" -czf "/tmp/benchsuite-$TAG.tgz" ops/benchsuite
scp "${SSHO[@]}" -i "$(pod_ssh_key "$POD")" -P "$(podf "$POD" ssh_port)" "/tmp/benchsuite-$TAG.tgz" "$(podf "$POD" ssh_user root)@$(podf "$POD" ssh_host):/tmp/benchsuite.tgz" || finish 4   # /tmp: writable for any login user
rm -f "/tmp/benchsuite-$TAG.tgz"
ssh_pod "mkdir -p /root/affine /root/benchsuite/runs && cd /root/affine && tar xzf /tmp/benchsuite.tgz && BENCH_HOME=/root/benchsuite bash /root/affine/ops/benchsuite/install_eval_env.sh > /root/install.log 2>&1; tail -1 /root/install.log" | tee "$RUN_DIR/capfill-install.log" | grep -q INSTALL_DONE || { log "install failed"; finish 5; }
[ -n "${DOCKERHUB_TOKEN:-}" ] && printf '%s' "$DOCKERHUB_TOKEN" | ssh_pod "docker login -u '$DOCKERHUB_USER' --password-stdin >/dev/null 2>&1" || true
ssh_pod "cd /root/affine/ops/benchsuite && export PATH=\$HOME/.local/bin:\$PATH && /root/benchsuite/verifiers/.venv/bin/python lock.py check --bench-home /root/benchsuite" || { log "LOCK MISMATCH"; finish 10; }
log "installed after $(( ($(date +%s) - T0) / 60 )) min; running $CAP_ENVS ($SIDE) at the new caps"

# ---- 4. the cells, into the SAME run id
URL="http://127.0.0.1:$(podf "$POD" front_internal)/v1"; SERVED="$(podf "$POD" served)"
if [ "$SIDE" = teacher ]; then MODELARGS="--teacher-url $URL --teacher-model $SERVED --models teacher"
else MODELARGS="--king-url $URL --king-model $SERVED --models king --teacher-from $TEACHER_FROM"; fi
PYR=/root/benchsuite/verifiers/.venv/bin/python
ssh_pod "export BENCH_API_KEY='$(podf "$POD" key)' PRIME_API_KEY='${PRIME_API_KEY:-}' HF_TOKEN='${HF_TOKEN:-}' BENCHSUITE_CHAT_IMAGE=affine-bench-chat:py311 BENCHSUITE_SETTINGS_JSON='${BENCHSUITE_SETTINGS_JSON:-}'; cd /root/affine/ops/benchsuite && $PYR run_suite.py run --run-id $RUN_ID --key-env BENCH_API_KEY $MODELARGS --verifiers-dir /root/benchsuite/verifiers --out /root/benchsuite/runs --runtime $RUNTIME --envs $CAP_ENVS --temps primary,secondary --concurrency 64 --parallel-envs $CAP_PARALLEL --manifest manifest-capfill.json --push" > "$RUN_DIR/capfill-run.log" 2>&1 &
RPID=$!
pull_light() { ssh_pod "cd /root/benchsuite/runs && tar czf - --exclude='*/logs' --exclude='*/traces.jsonl*' --exclude='*/eval.log' --exclude='manifest.json' $RUN_ID 2>/dev/null" 2>/dev/null | tar xzf - -C "$BENCH_HOME/runs" 2>/dev/null; }
while kill -0 "$RPID" 2>/dev/null; do
  sleep 300; pull_light
  "$PY" "$HERE/publish.py" --run-dir "$RUN_DIR" --only-state --partial --eta >/dev/null 2>&1 || true
done
wait "$RPID"; RC=$?
log "cells ended (exit $RC) after $(( ($(date +%s) - T0) / 60 )) min"

# ---- 5. tail: full pull, infra retry, publish (uploads the new cells and the re-tagged old ones)
ssh_pod "cd /root/benchsuite/runs && tar czf - --exclude='*/logs/attempt_*' --exclude='manifest.json' $RUN_ID 2>/dev/null" 2>/dev/null | tar xzf - -C "$BENCH_HOME/runs" 2>/dev/null
ssh_pod "export BENCH_API_KEY='$(podf "$POD" key)' BENCHSUITE_CHAT_IMAGE=affine-bench-chat:py311; cd /root/affine/ops/benchsuite && $PYR run_suite.py retry --run-id $RUN_ID --out /root/benchsuite/runs --verifiers-dir /root/benchsuite/verifiers" > "$RUN_DIR/capfill-retry.log" 2>&1
ssh_pod "cd /root/benchsuite/runs && tar czf - --exclude='*/logs/attempt_*' --exclude='manifest.json' $RUN_ID 2>/dev/null" 2>/dev/null | tar xzf - -C "$BENCH_HOME/runs" 2>/dev/null
"$PY" "$HERE/run_suite.py" summarize --run-id "$RUN_ID" --out "$BENCH_HOME/runs" >/dev/null 2>&1 || true
"$PY" - "$RUN_DIR/manifest.json" "$CAP_ENVS" "$SIDE" "$POD" "$(( ($(date +%s) - T0) / 60 ))" <<'PY'
import json, sys, time
p, envs, side, pod, mins = sys.argv[1:]
m = json.load(open(p))
m.setdefault("cap_backfill", []).append({"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "side": side, "envs": envs.split(","),
                                          "pod": pod, "minutes": int(mins), "note": "cells re-run at the 2026-09-19 completion caps; old-cap cells kept as <env>@<cap>k"})
json.dump(m, open(p, "w"), indent=1)
PY
[ "$SIDE" = king ] && "$PY" "$HERE/recap.py" --run-dir "$RUN_DIR" --teacher-from "$TEACHER_FROM" >/dev/null 2>&1
"$PY" "$HERE/publish.py" --run-dir "$RUN_DIR" || finish 8
log "done: $RUN_ID ($SIDE) at the new caps in $(( ($(date +%s) - T0) / 60 )) min"
finish 0
