#!/bin/bash
# One benchmark-suite pass, end to end, on Prime Intellect's stack.
#
#   run_pass.sh <model_ref> <label> <run_id> [MODE]        MODE defaults to [modes].default
#
#   lium         DEFAULT (operator decision 2026-09-13: in-house GPUs, paid in TAO).
#                <model_ref> = king digest, <label> = reign. King on a Lium pod rented for
#                the run (kingpod.py -> ops/king-datagen/bootstrap_king.sh: the king-seat
#                stack, vLLM 0.28.0, qwen3 parsers), the eval driver on the same pod
#                (Lium DinD templates ship docker), teacher reused from [modes].teacher_from,
#                the SAME cells/seeds/caps as the Prime runs (suite.lock.json is checked on
#                the pod before the first cell). Chat sets always; sandbox sets under the
#                docker runtime on the pod when a chat cell moved (or BENCHSUITE_FORCE_SANDBOX=1).
#   prime        <model_ref> = king sha256 digest (public models.affine.io copy),
#                <label> = reign number. King only on a Prime pod (4 TP2 replicas),
#                teacher reused from [modes].teacher_from; chat sets always,
#                sandbox sets when a chat cell moved vs the previous published run
#                (or BENCHSUITE_FORCE_SANDBOX=1). The standing per-crown / weekly mode.
#   full         same pod, king + teacher served (2 TP2 each), every env, no gate.
#                The reference pass that refreshes the teacher baseline.
#   challenger   <model_ref> = r2://affine-private-models/models/registrations/<reg>/
#                (+ CHALLENGER_REVISION=<sha256>, CHALLENGER_MARGIN/Z/VS_REIGN for the card),
#                <label> = chal-NNNNN. Lium 1x H200 (king-seat bootstrap reading the private
#                bucket with the eval pods' read-only key), chat sets only, teacher reused.
#   genesis      hf://Qwen/Qwen3.6-35B-A3B@<rev> on a Lium 1x H200 (bootstrap_king.sh HF path), chat sets
#                only, teacher reused — the unpaid reign-0 baseline row (2026-09-15).
#   comparables  <model_ref> = Prime Inference model id (e.g. qwen/qwen3.6-35b-a3b),
#                <label> = display label. No pod: the eval driver runs here (docker),
#                the model is Prime Inference, chat sets only, teacher reused.
#   cheap        fallback: king on a Lium 1x H200 (kingpod.py), chat sets, teacher
#                reused; sandbox sets on Prime only when a chat cell moved.
#
# Env (ops/benchsuite/run.sh loads the box snapshot): PRIME_API_KEY (or PRIME),
# HF_TOKEN, DATA_R2_*, LIUM_API_KEY (cheap), AFFINE_EVAL_R2_* (challenger),
# PRIME_SSH_KEY (default ~/.ssh/prime_bench, registered on the Prime account).
# Docker Hub pull-cap login for the pod: DOCKERHUB_USER / DOCKERHUB_TOKEN, or —
# when unset — read at pod time from the Arbos vault (op CLI +
# OP_SERVICE_ACCOUNT_TOKEN from ~/.affine-op.env; item DOCKERHUB_OP_ITEM).
# Exit code -> ops/benchsuite/state/pass-<run_id>.exit for watch.py.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="${BENCHSUITE_PYTHON:-$REPO/.venv/bin/python}"
BENCH_HOME="${BENCH_HOME:-$HOME/benchsuite}"
REF="$1"; LABEL="$2"; RUN_ID="$3"
toml() { "$PY" -c 'import tomllib,sys; d=tomllib.load(open("'"$HERE"'/suite.toml","rb")); v=d
for k in sys.argv[1].split("."): v=v[k]
print(",".join(v) if isinstance(v,list) else v)' "$1"; }
MODE="${4:-${BENCHSUITE_MODE:-$(toml modes.default)}}"
mkdir -p "$HERE/state"
LOG_EXIT="$HERE/state/pass-$RUN_ID.exit"
echo $$ > "$HERE/state/pass-$RUN_ID.pid"   # watch.py polls this (the driver is detached from pm2's tree)
log() { echo "[run_pass] $(date -u +%FT%TZ) $*"; }
finish() { echo "$1" > "$LOG_EXIT"; exit "$1"; }
CODE_COMMIT=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)
RUN_DIR="$BENCH_HOME/runs/$RUN_ID"
mkdir -p "$RUN_DIR"
export PRIME_API_KEY="${PRIME_API_KEY:-${PRIME:-}}"
TEACHER_FROM=$(toml modes.teacher_from)
# BENCHSUITE_CHAT_ENVS (comma list) restricts the chat cells — used to add one
# new env to an existing card (When2Call on reign 11) without re-running the rest.
CHAT_ENVS="${BENCHSUITE_CHAT_ENVS:-$(toml modes.chat_envs)}"
# `agentic` mode runs modes.agentic_envs (Terminal-Bench 2, tau2, SWE-bench Pro) as
# the "chat" phase on its own pod — docker for the Harbor sets, subprocess for tau2 —
# and merges the cells into an existing card (BENCHSUITE_MERGE_INTO=<run_id>,
# BENCHSUITE_MERGE_AS=king|teacher) instead of publishing a card of its own.
[ "$MODE" = "agentic" ] && CHAT_ENVS="${BENCHSUITE_CHAT_ENVS:-$(toml modes.agentic_envs)}"
SANDBOX_ENVS=$(toml modes.sandbox_envs)
PREV_CARD=$(ls -t "$REPO/$(toml suite.state_dir)"/*.json 2>/dev/null | head -1)

# Docker Hub login on the pod. Measured 2026-09-13: anonymous = 100 pulls/h per IP,
# the thebes1618 login = 200 pulls/h per user (free plan); a SWE-bench pass pulls
# 500 images. Values come from DOCKERHUB_USER / DOCKERHUB_TOKEN or, when unset, from
# the Arbos vault item (Login "Docker Hub — thebes1618 (pull-cap login for Lium
# bench/datagen pods)"). The token goes to the pod over ssh stdin, never on a
# command line (no `ps` exposure on the box or the pod).
DOCKERHUB_OP_ITEM="${DOCKERHUB_OP_ITEM:-op://Arbos/e7y3qzb2flaczam4rindajho4m}"
dockerhub_creds() {
  if [ -z "${DOCKERHUB_TOKEN:-}" ] && [ -n "${OP_SERVICE_ACCOUNT_TOKEN:-}" ] && command -v op >/dev/null 2>&1; then
    DOCKERHUB_USER=$(op read --no-newline "$DOCKERHUB_OP_ITEM/username" 2>/dev/null) || DOCKERHUB_USER=""
    DOCKERHUB_TOKEN=$(op read --no-newline "$DOCKERHUB_OP_ITEM/credential" 2>/dev/null) || DOCKERHUB_TOKEN=""
    [ -n "$DOCKERHUB_TOKEN" ] && log "docker hub credential read from the vault (user ${DOCKERHUB_USER})"
  fi
  [ -n "${DOCKERHUB_USER:-}" ] && [ -n "${DOCKERHUB_TOKEN:-}" ]
}
dockerhub_login() {  # $1 = sudo prefix ("" or "sudo "); uses the caller's SSH array
  if ! dockerhub_creds; then log "no docker hub credential; anonymous pulls (100/h per IP)"; return 0; fi
  if printf '%s' "$DOCKERHUB_TOKEN" | "${SSH[@]}" "$1docker login -u '$DOCKERHUB_USER' --password-stdin >/dev/null 2>&1"; then
    log "docker hub login ok on the pod (user $DOCKERHUB_USER)"
  else
    log "docker hub login failed; anonymous pulls"
  fi
}

# The eval driver, run either here or on the pod (RUNNER="ssh ..." prefix).
# $1 = extra run_suite.py flags, $2 = runtime, $3 = envs, $4 = temps, $5 = concurrency, $6 = manifest
suite_cmd() {  # $7 = parallel envs (default 2; the sandbox phase uses 1 so SWE-bench runs alone with its fixed budget)
  echo "run_suite.py run --run-id $RUN_ID --key-env BENCH_API_KEY --pod-usd-per-hour ${USD_HR:-0} --meta meta.json $1 --runtime $2 --envs $3 --temps $4 --concurrency $5 --parallel-envs ${7:-2} --manifest $6"
}

# Sandbox gate: did a king chat cell move vs the newest published scorecard?
sandbox_trigger() {
  if [ "${BENCHSUITE_FORCE_SANDBOX:-0}" = "1" ]; then echo flag; return; fi
  if [ -z "$PREV_CARD" ]; then echo no-baseline; return; fi
  if "$PY" "$HERE/run_suite.py" compare --run-id "$RUN_ID" --out "$BENCH_HOME/runs" --against "$PREV_CARD" > "$RUN_DIR/compare.txt" 2>&1; then echo moved; else echo none; fi
}
stamp_trigger() {  # trigger
  "$PY" - "$RUN_DIR/manifest.json" "$1" "${PREV_CARD:-}" <<'PY'
import json, sys
p, trig, prev = sys.argv[1:]
m = json.load(open(p)); m["sandbox_trigger"] = {"trigger": trig, "compared_against": prev}; json.dump(m, open(p, "w"), indent=1)
PY
}

# ===================================================== Prime pod passes ======
# prime / full / challenger share this: rent a pod, serve the model(s), run.
run_on_prime_pod() {  # $1 = models (king|king,teacher)  $2 = sandbox policy (always|gated|never)
  local MODELS="$1" SANDBOX_POLICY="$2" POD_ID=""
  local SSH_KEY="${PRIME_SSH_KEY:-$HOME/.ssh/prime_bench}" KH="$HERE/state/prime_known_hosts"
  cleanup_pod() { [ -n "$POD_ID" ] && { log "terminating Prime pod $POD_ID"; "$PY" "$HERE/primepod.py" terminate "$POD_ID" || true; }; }
  trap cleanup_pod EXIT
  local CREATE; CREATE=$("$PY" "$HERE/primepod.py" create --name "affine-benchsuite-$RUN_ID" | tail -1) || finish 2
  POD_ID=$(echo "$CREATE" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["pod_id"])')
  USD_HR=$(echo "$CREATE" | "$PY" -c 'import json,sys; print(json.load(sys.stdin).get("usd_per_hour") or 0)')
  local SSH_TARGET; SSH_TARGET=$("$PY" "$HERE/primepod.py" wait "$POD_ID" --timeout-min 40) || finish 3
  local USER_HOST=${SSH_TARGET% *} PORT=${SSH_TARGET##* }
  SSH=(ssh -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=20 -o LogLevel=ERROR -p "$PORT" "$USER_HOST")
  SCP=(scp -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o LogLevel=ERROR -P "$PORT")
  log "pod $POD_ID up at $USER_HOST:$PORT (\$$USD_HR/h) models=$MODELS"

  # serving env
  local API_KEY; API_KEY=$("$PY" -c 'import secrets; print(secrets.token_hex(24))')
  local KING_REPLICAS TEACHER_REPLICAS KING_LINE
  if [ "$MODELS" = "king,teacher" ]; then
    KING_REPLICAS=$(toml prime.king_replicas); TEACHER_REPLICAS=$(toml prime.teacher_replicas)
  else
    KING_REPLICAS=$(toml modes.prime_king_replicas); TEACHER_REPLICAS=""
  fi
  if [[ "$REF" == r2://* ]]; then
    KING_LINE="KING_R2=\"$REF\"
KING_DIGEST=\"${CHALLENGER_REVISION:?CHALLENGER_REVISION (sha256 model_digest) required for an r2:// ref}\"
AFFINE_EVAL_R2_ENDPOINT=\"${AFFINE_EVAL_R2_ENDPOINT:?}\"
AFFINE_EVAL_R2_ACCESS_KEY_ID=\"${AFFINE_EVAL_R2_ACCESS_KEY_ID:?}\"
AFFINE_EVAL_R2_SECRET_ACCESS_KEY=\"${AFFINE_EVAL_R2_SECRET_ACCESS_KEY:?}\""
  else
    KING_LINE="KING_DIGEST=\"$REF\""
  fi
  local ENV_TMP; ENV_TMP=$(mktemp /tmp/benchenv.XXXXXX)
  cat > "$ENV_TMP" <<EOF
$KING_LINE
TEACHER_HF="$(toml teacher.hf_repo)"
HF_TOKEN="${HF_TOKEN:-}"
KING_REPLICAS="$KING_REPLICAS"
TEACHER_REPLICAS="$TEACHER_REPLICAS"
API_KEY="$API_KEY"
MAX_MODEL_LEN="$(toml serving.max_model_len)"
GPU_UTIL="$(toml serving.gpu_memory_utilization)"
BATCHED_TOKENS="$(toml serving.max_num_batched_tokens)"
MAX_NUM_SEQS="$(toml serving.max_num_seqs)"
VLLM_VERSION="$(toml serving.vllm_version)"
VLLM_CUDA="${VLLM_CUDA:-cu129}"
EOF
  "${SCP[@]}" "$ENV_TMP" "$HERE/pod_bootstrap.sh" "$USER_HOST:/tmp/" || finish 4
  local ENV_BN; ENV_BN=$(basename "$ENV_TMP"); rm -f "$ENV_TMP"
  "${SSH[@]}" "sudo mkdir -p /root/bench /root/logs && sudo mv /tmp/$ENV_BN /root/bench/env && sudo chmod 600 /root/bench/env && sudo mv /tmp/pod_bootstrap.sh /root/bench/bootstrap.sh && sudo bash -c 'cd /root && HOME=/root nohup setsid bash /root/bench/bootstrap.sh >> /root/bench/bootstrap.log 2>&1 < /dev/null &'; sudo usermod -aG docker \$USER 2>/dev/null; echo launched" || finish 4

  # wait for every replica (bootstrap writes /root/bench/ready)
  for _ in $(seq 1 120); do
    "${SSH[@]}" 'sudo test -f /root/bench/ready' 2>/dev/null && break
    if "${SSH[@]}" 'sudo test -f /root/bench/bootstrap.failed' 2>/dev/null; then log "bootstrap failed: $("${SSH[@]}" 'sudo cat /root/bench/bootstrap.failed')"; finish 6; fi
    sleep 30
  done
  "${SSH[@]}" 'sudo test -f /root/bench/ready' || { log "serving never became ready"; finish 6; }
  log "serving ready ($MODELS)"
  remote_suite "$MODELS" "$SANDBOX_POLICY" prime "$USER_HOST" "$PORT" "$SSH_KEY" "$KH" "$API_KEY" \
    "$USD_HR" "$POD_ID" "http://127.0.0.1:8001/v1" king "http://127.0.0.1:8002/v1" teacher "Prime Intellect pod"
  finish 0
}

# Install the eval env on a remote pod, copy the teacher baseline, run the chat cells,
# apply the sandbox gate, run the sandbox cells (runtime $3: prime|docker), retry infra
# errors, pull the run back and publish. Used by the Prime and the Lium paths.
# Daytona (Harbor cloud sandboxes): key from the env or the Arbos vault item "Daytona Arbos".
DAYTONA_OP_ITEM="${DAYTONA_OP_ITEM:-op://Arbos/fywmj6vtq5delybw5c7a53l2qa/notesPlain}"
daytona_key() {
  if [ -z "${DAYTONA_API_KEY:-}" ] && [ -n "${OP_SERVICE_ACCOUNT_TOKEN:-}" ] && command -v op >/dev/null 2>&1; then
    DAYTONA_API_KEY=$(op read --no-newline "$DAYTONA_OP_ITEM" 2>/dev/null | grep -o 'dtn_[A-Za-z0-9_-]*' | head -1) || DAYTONA_API_KEY=""
    export DAYTONA_API_KEY
    [ -n "$DAYTONA_API_KEY" ] && log "daytona key read from the vault"
  fi
  [ -n "${DAYTONA_API_KEY:-}" ] && [ -x "${HARBOR_BIN:-$HOME/benchsuite/harborenv/bin/harbor}" ]
}

remote_suite() {  # models policy sandbox_runtime user@host port key known_hosts api_key usd_hr pod_id king_url king_model teacher_url teacher_model provider
  local MODELS="$1" SANDBOX_POLICY="$2" SB_RUNTIME="$3" USER_HOST="$4" PORT="$5" SSH_KEY="$6" KH="$7"
  local API_KEY="$8" USD_HR="$9" POD_ID="${10}" KING_URL="${11}" KING_MODEL="${12}" TEACHER_URL="${13}" TEACHER_MODEL="${14}" PROVIDER="${15}"
  SSH=(ssh -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=20 -o LogLevel=ERROR -p "$PORT" "$USER_HOST")
  SCP=(scp -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o LogLevel=ERROR -P "$PORT")
  local RHOME; RHOME=$("${SSH[@]}" 'echo $HOME')
  local SUDO=""; [ "$USER_HOST" = "${USER_HOST#root@}" ] && SUDO="sudo "
  tar -C "$REPO" -czf "/tmp/benchsuite-$RUN_ID.tgz" ops/benchsuite
  "${SCP[@]}" "/tmp/benchsuite-$RUN_ID.tgz" "$USER_HOST:/tmp/benchsuite.tgz" || finish 4
  rm -f "/tmp/benchsuite-$RUN_ID.tgz"
  "${SSH[@]}" "mkdir -p $RHOME/affine $RHOME/benchsuite/runs && cd $RHOME/affine && tar xzf /tmp/benchsuite.tgz && BENCH_HOME=$RHOME/benchsuite bash $RHOME/affine/ops/benchsuite/install_eval_env.sh > $RHOME/install.log 2>&1; tail -1 $RHOME/install.log" || finish 5
  "${SSH[@]}" "${SUDO}usermod -aG docker \$USER 2>/dev/null; true"
  dockerhub_login "$SUDO"
  "${SSH[@]}" "${SUDO}docker pull -q python:3.11-slim >/dev/null 2>&1; true"
  # docker preflight (2026-09-17): the Genesis sandbox pass burnt 3 h and published
  # 0.0 for SWE-bench after `docker run` failed on every task ("error creating
  # overlay mount ... function not implemented" on that executor); the Occamy pass
  # lost SWE-bench to "toomanyrequests" on the first pull. Fail here, fast, with a
  # named cause, instead of erroring 500 rollouts at 0 tokens.
  local DOCK; DOCK=$("${SSH[@]}" "${SUDO}docker run --rm python:3.11-slim true >/dev/null 2>&1 && echo RUN_OK || echo RUN_FAIL; ${SUDO}docker system info 2>/dev/null | grep -q 'Username:' && echo AUTH_OK || echo AUTH_ANON" 2>/dev/null | tr '\n' ' ')
  if [[ "$DOCK" != *RUN_OK* ]]; then
    log "DOCKER PREFLIGHT FAILED: this executor cannot run containers ($DOCK) — striking it and giving the pod back"
    [ -n "${POD_ID:-}" ] && "$PY" "$HERE/kingpod.py" release "$POD_ID" --strike docker_cannot_run >/dev/null 2>&1 && trap - EXIT
    finish 7
  fi
  [[ "$DOCK" == *AUTH_ANON* ]] && log "WARNING: docker hub login did not stick ($DOCK); pulls are anonymous (100/h per IP)"
  # the lock: the pod's env must match suite.lock.json (code commits, patches, grader packages, serving)
  # BENCHSUITE_LOCK_WRITE=1 is the deliberate way to re-pin (new env, new patch):
  # the pod writes suite.lock.json from what install_eval_env.sh produced and the
  # box copy is replaced; the diff is logged and the file is then committed.
  if ! "${SSH[@]}" "cd $RHOME/affine/ops/benchsuite && $RHOME/benchsuite/verifiers/.venv/bin/python lock.py check --bench-home $RHOME/benchsuite"; then
    if [ "${BENCHSUITE_LOCK_WRITE:-0}" = "1" ]; then
      log "LOCK MISMATCH — BENCHSUITE_LOCK_WRITE=1: re-pinning suite.lock.json from the pod"
      "${SSH[@]}" "cd $RHOME/affine/ops/benchsuite && $RHOME/benchsuite/verifiers/.venv/bin/python lock.py write --bench-home $RHOME/benchsuite" || finish 10
      "${SCP[@]}" "$USER_HOST:$RHOME/affine/ops/benchsuite/suite.lock.json" "$HERE/suite.lock.json" || finish 10
    else
      log "LOCK MISMATCH on the pod — refusing to run"; finish 10
    fi
  fi
  if [ "$MODELS" = "king" ] && [ -d "$BENCH_HOME/runs/$TEACHER_FROM" ]; then
    tar -C "$BENCH_HOME/runs" -czf "/tmp/teacher-$RUN_ID.tgz" --exclude 'traces.jsonl*' --exclude 'logs' "$TEACHER_FROM/teacher"
    "${SCP[@]}" "/tmp/teacher-$RUN_ID.tgz" "$USER_HOST:/tmp/teacher.tgz" && "${SSH[@]}" "cd $RHOME/benchsuite/runs && tar xzf /tmp/teacher.tgz"
    rm -f "/tmp/teacher-$RUN_ID.tgz"
  fi
  # hardware of the serving pod (Lium: pods.json machine / plan), recorded on the card
  local POD_GPU="" POD_PLAN=""
  if [ -n "$POD_ID" ] && [ -f "$HERE/state/pods.json" ]; then
    POD_GPU=$("$PY" -c 'import json,sys; m=json.load(open(sys.argv[1])).get(sys.argv[2]) or {}; print(m.get("machine") or "")' "$HERE/state/pods.json" "$POD_ID" 2>/dev/null || echo "")
    POD_PLAN=$("$PY" -c 'import json,sys; m=json.load(open(sys.argv[1])).get(sys.argv[2]) or {}; print((m.get("plan") or {}).get("name") or "")' "$HERE/state/pods.json" "$POD_ID" 2>/dev/null || echo "")
  fi
  local META; META=$(BENCH_POD_GPU="$POD_GPU" BENCH_POD_PLAN="$POD_PLAN" "$PY" - "$REF" "$LABEL" "$POD_ID" "$USD_HR" "$CODE_COMMIT" "$MODE" "$TEACHER_FROM" "$MODELS" "$PROVIDER" "$SB_RUNTIME" <<'PY'
import json, os, sys
ref, label, pod, usd, commit, mode, tfrom, models, provider, sbr = sys.argv[1:]
if ref.startswith("r2://"):
    king = {"repo": ref, "digest": os.environ.get("CHALLENGER_REVISION", "")}
elif ref.startswith("hf://"):
    spec = ref[len("hf://"):]
    king = {"repo": ref, "hf_repo": spec.split("@")[0], "hf_revision": spec.partition("@")[2],
            "digest": "hf-" + spec.partition("@")[2][:10]}
else:
    king = {"digest": ref}
if label.isdigit(): king["reign"] = int(label)
else: king["label"] = label
duel = {k: os.environ.get(f"CHALLENGER_{k.upper()}") for k in ("margin", "z", "vs_reign", "vs_king_digest", "judged_at", "hotkey")}
if any(duel.values()):
    king["duel"] = {k: (float(v) if k in ("margin", "z") and v not in (None, "") else v) for k, v in duel.items()}
meta = {"mode": mode, "king": king,
        "where": {"provider": provider, "pod_id": pod, "usd_per_hour": float(usd),
                  "eval_driver": f"same pod; docker runtime for chat sets, {sbr} runtime for sandbox sets"},
        "code": {"affine_commit": commit}}
if os.environ.get("BENCH_POD_GPU"):
    meta["where"]["gpu"] = os.environ["BENCH_POD_GPU"]
if os.environ.get("BENCH_POD_PLAN"):
    meta["where"]["plan"] = os.environ["BENCH_POD_PLAN"]
if models == "king": meta["teacher"] = {"reused_from": tfrom}
print(json.dumps(meta))
PY
)
  local MODEL_FLAGS="--king-url $KING_URL --king-model $KING_MODEL --models $MODELS --verifiers-dir $RHOME/benchsuite/verifiers --out $RHOME/benchsuite/runs --push"
  [ "$MODELS" = "king,teacher" ] && MODEL_FLAGS="$MODEL_FLAGS --teacher-url $TEACHER_URL --teacher-model $TEACHER_MODEL" || MODEL_FLAGS="$MODEL_FLAGS --teacher-from $TEACHER_FROM"
  local REMOTE_ENV="export BENCH_API_KEY='$API_KEY' PRIME_API_KEY='${PRIME_API_KEY:-}' HF_TOKEN='${HF_TOKEN:-}' BENCHSUITE_CHAT_IMAGE=affine-bench-chat:py311; cd $RHOME/affine/ops/benchsuite && echo '$META' > meta.json"
  local PYR="$RHOME/benchsuite/verifiers/.venv/bin/python"
  # cells publish as they finish (Jacob 2026-09-17): while the chat suite runs on the pod,
  # pull the light files (summaries, manifests, cmd.txt — no traces) every 10 min and
  # republish the partial card; the final publish at the end is the same operation.
  "${SSH[@]}" "$REMOTE_ENV && $PYR $(suite_cmd "$MODEL_FLAGS" docker "$CHAT_ENVS" primary,secondary 64 manifest.json)" > "$RUN_DIR/chat-suite.log" 2>&1 &
  local CHAT_PID=$!
  while kill -0 "$CHAT_PID" 2>/dev/null; do
    sleep 600
    kill -0 "$CHAT_PID" 2>/dev/null || break
    pull_light "$USER_HOST" "$PORT" "$SSH_KEY" "$KH" "$RHOME" && [ -z "${BENCHSUITE_MERGE_INTO:-}" ] && { "$PY" "$HERE/publish.py" --run-dir "$RUN_DIR" --only-state --partial >/dev/null 2>&1 || true; }
  done
  wait "$CHAT_PID" || log "chat suite returned non-zero; continuing"
  cat "$RUN_DIR/chat-suite.log"
  pull_run "$USER_HOST" "$PORT" "$SSH_KEY" "$KH" "$RHOME"
  # PARTIAL card now (chat cells): the attribution job watches affine/state/benchsuite/*.json
  local PARTIAL_FLAG=""; [ "$SANDBOX_POLICY" != "never" ] && PARTIAL_FLAG="--partial"
  # the chat cells go to R2 right here (traces included), not only at the end of the pass:
  # a driver that dies later must not leave them box-only (reign 13, 2026-09-15..17)
  [ -z "${BENCHSUITE_MERGE_INTO:-}" ] && { "$PY" "$HERE/publish.py" --run-dir "$RUN_DIR" --only-cells "king/*,teacher/*" $PARTIAL_FLAG || log "partial publish failed; continuing"; }
  local TRIGGER="always"
  [ "$SANDBOX_POLICY" = "gated" ] && TRIGGER=$(sandbox_trigger)
  [ "$SANDBOX_POLICY" = "never" ] && TRIGGER="never"
  stamp_trigger "$TRIGGER"
  if [ "$TRIGGER" != "none" ] && [ "$TRIGGER" != "never" ]; then
    if [ "$SB_RUNTIME" = "docker" ]; then
      # Lium: docker on the pod for the public-image sets, Prime sandboxes for the Lean set.
      # BENCHSUITE_SANDBOX=daytona (2026-09-17): the Harbor-able sets (sandbox_daytona.envs)
      # run through Harbor on Daytona from the BOX against the pod's public endpoint, in
      # parallel with the pod's remaining docker cells; the pod only serves the model.
      local DOCKER_SB_ENVS; DOCKER_SB_ENVS=$(toml modes.lium_docker_sandbox_envs | tr "," " ")
      local DAYTONA_PID=""
      if [ "${BENCHSUITE_SANDBOX:-}" = "daytona" ]; then
        local DT_ENVS; DT_ENVS=$(toml sandbox_daytona.envs | tr "," " ")
        local DT_RUN=""; DOCKER_SB_ENVS=""
        for SB_ENV in $(toml modes.lium_docker_sandbox_envs | tr "," " "); do
          if [[ " $DT_ENVS " == *" $SB_ENV "* ]]; then DT_RUN="$DT_RUN $SB_ENV"; else DOCKER_SB_ENVS="$DOCKER_SB_ENVS $SB_ENV"; fi
        done
        local EXT_URL; EXT_URL=$("$PY" -c 'import json; print(json.load(open("'"$HERE"'/state/pods.json"))["'"$POD_ID"'"]["base_url"])' 2>/dev/null || echo "")
        if [ -n "$DT_RUN" ] && [ -n "$EXT_URL" ] && daytona_key; then
          log "sandbox sets ($TRIGGER): Harbor on Daytona from the box for [$DT_RUN] (pod endpoint $EXT_URL); docker on the pod for [$DOCKER_SB_ENVS]"
          ( export BENCH_API_KEY="$API_KEY"
            for DT_ENV in $DT_RUN; do
              "$PY" "$HERE/harbor_cell.py" run --env "$DT_ENV" --model "$KING_MODEL" --model-label king --model-url "$EXT_URL" --model-key-env BENCH_API_KEY \
                --out "$RUN_DIR/king" --concurrency "$(toml sandbox_daytona.concurrency)" --agent-timeout-s "$(toml sandbox_daytona.budgets.default.agent_timeout_s)" \
                --step-limit "$(toml sandbox_daytona.budgets.default.step_limit)" || log "daytona cell $DT_ENV returned non-zero; continuing"
            done ) > "$RUN_DIR/daytona.log" 2>&1 &
          DAYTONA_PID=$!
        else
          log "BENCHSUITE_SANDBOX=daytona but no Daytona key / pod url / Harbor-able env; falling back to docker on the pod"; DOCKER_SB_ENVS=$(toml modes.lium_docker_sandbox_envs | tr "," " ")
        fi
      fi
      log "sandbox sets ($TRIGGER): docker on the pod for [$DOCKER_SB_ENVS]; Prime sandboxes for $(toml modes.lium_prime_sandbox_envs)"
      local SB_ENV SB_FAIL=""
      for SB_ENV in $DOCKER_SB_ENVS; do
        "${SSH[@]}" "$REMOTE_ENV && $PYR $(suite_cmd "$MODEL_FLAGS" docker "$SB_ENV" primary 48 manifest-sandbox.json 1)" || log "docker sandbox cell $SB_ENV returned non-zero; continuing"
        # fail fast: a cell whose rollouts ALL errored is an infrastructure failure
        # (docker cannot run / Docker Hub rate limit); the remaining docker cells
        # would burn the same way, so stop here and let the queue retry later.
        local VERDICT; VERDICT=$("${SSH[@]}" "$PYR - <<'PY'
import json, glob, collections
paths = glob.glob('$RHOME/benchsuite/runs/$RUN_ID/king/${SB_ENV}__t0/summary.json')
if not paths:
    print('nosummary'); raise SystemExit
s = json.load(open(paths[0])); n = int(s.get('n') or 0); e = int(s.get('n_errored') or 0)
kind = 'ok'
if n and e >= n:
    kind = 'allerrored'
    try:
        c = collections.Counter()
        for line in open(paths[0].replace('summary.json', 'traces.jsonl')):
            t = json.loads(line); tr = t.get('trace') or t
            for err in (tr.get('errors') or [])[:1]:
                m = err.get('message', '')
                c['ratelimit' if 'toomanyrequests' in m else 'overlay' if 'overlay mount' in m else 'other'] += 1
        kind += ':' + (c.most_common(1)[0][0] if c else 'unknown')
    except Exception:
        kind += ':unknown'
print(kind, n, e)
PY" 2>/dev/null)
        log "docker cell $SB_ENV verdict: $VERDICT"
        pull_light "$USER_HOST" "$PORT" "$SSH_KEY" "$KH" "$RHOME" && [ -z "${BENCHSUITE_MERGE_INTO:-}" ] && { "$PY" "$HERE/publish.py" --run-dir "$RUN_DIR" --only-state --partial >/dev/null 2>&1 || true; }
        if [[ "$VERDICT" == allerrored* ]]; then SB_FAIL="$VERDICT"; log "ABORTING the remaining docker sandbox cells: $SB_ENV failed as a run ($VERDICT)"; break; fi
      done
      "${SSH[@]}" "$REMOTE_ENV && $PYR $(suite_cmd "$MODEL_FLAGS" prime "$(toml modes.lium_prime_sandbox_envs)" primary 32 manifest-sandbox-prime.json)" || log "prime sandbox suite returned non-zero; continuing"
      if [ -n "$DAYTONA_PID" ]; then log "waiting for the Daytona cells (pid $DAYTONA_PID)"; wait "$DAYTONA_PID" || true; tail -3 "$RUN_DIR/daytona.log"; fi
    else
      log "sandbox sets ($TRIGGER) on Prime sandboxes"
      "${SSH[@]}" "$REMOTE_ENV && $PYR $(suite_cmd "$MODEL_FLAGS" prime "$SANDBOX_ENVS" primary 48 manifest-sandbox.json 1)" || log "sandbox suite returned non-zero; continuing"
    fi
  else
    log "no king chat cell moved beyond the previous run's interval; sandbox sets skipped"
  fi
  "${SSH[@]}" "$REMOTE_ENV && $PYR run_suite.py retry --run-id $RUN_ID --out $RHOME/benchsuite/runs --verifiers-dir $RHOME/benchsuite/verifiers && $PYR run_suite.py summarize --run-id $RUN_ID --out $RHOME/benchsuite/runs" || log "retry/summarize returned non-zero"
  # Prime Evals: the live --push streams while cells run; this replays anything it missed
  # (one platform run per cell, account `arbos`) and records the URLs in the summaries/manifest.
  "${SSH[@]}" "$REMOTE_ENV && $PYR push_evals.py --run-dir $RHOME/benchsuite/runs/$RUN_ID --models king" || log "push_evals returned non-zero; continuing"
  pull_run "$USER_HOST" "$PORT" "$SSH_KEY" "$KH" "$RHOME"
  if [ -n "${BENCHSUITE_MERGE_INTO:-}" ]; then merge_into_card; else "$PY" "$HERE/publish.py" --run-dir "$RUN_DIR" || finish 8; fi
  if [ -n "${SB_FAIL:-}" ]; then
    log "pass published with failed docker cells ($SB_FAIL): exiting 12 so the queue reruns them"
    finish 12
  fi
}

# Copy this run's king cells into another run's card (as king or teacher cells)
# and republish only those cells: the agentic set lands on the king's existing
# card instead of a second card for the same king.
merge_into_card() {
  local INTO="$BENCH_HOME/runs/${BENCHSUITE_MERGE_INTO}" AS="${BENCHSUITE_MERGE_AS:-king}" CELLS=""
  [ -d "$INTO" ] || { log "merge target $INTO missing"; finish 8; }
  for d in "$RUN_DIR"/king/*/; do
    [ -f "$d/summary.json" ] || continue
    local cell; cell=$(basename "$d")
    rm -rf "$INTO/$AS/$cell"; mkdir -p "$INTO/$AS"; cp -r "$d" "$INTO/$AS/$cell"
    "$PY" - "$INTO/$AS/$cell/summary.json" "$RUN_ID" "$(toml modes.lium_plan)" "$AS" <<'PY'
import json, sys
p, run_id, plan = sys.argv[1:]
s = json.load(open(p))
s["where"] = {"note": f"cell from agentic pass {run_id} (Lium {plan}, same lock)", "run_id": run_id, "provider": "Lium (our fleet, TAO)"}
# the pass ran the model as "king"; the card side it lands on is the merge target (2026-09-17: the
# teacher's agentic cells published as King 11's because publish.py keys cells by summary["model"])
if len(sys.argv) > 4:
    s["model"] = sys.argv[4]
json.dump(s, open(p, "w"), indent=1)
PY
    CELLS="${CELLS:+$CELLS,}$AS/$cell"
  done
  cp "$RUN_DIR/manifest.json" "$INTO/manifest-agentic-$AS.json" 2>/dev/null || true
  log "merged cells [$CELLS] into $BENCHSUITE_MERGE_INTO as $AS"
  "$PY" "$HERE/publish.py" --run-dir "$INTO" --only-cells "$CELLS" || finish 8
}

# Re-attach to a Lium pod whose box-side driver died (2026-09-15: a pm2 restart
# of the watcher killed the reign-13 driver while the pod kept running its cells).
# Waits for the pod's run_suite processes to end, then does the tail the driver
# would have done: retry + summarize, Prime Evals push, pull, publish, release.
#   BENCHSUITE_ATTACH_POD=<pod name> run_pass.sh <ref> <label> <run_id> attach
attach_lium() {
  local POD="${BENCHSUITE_ATTACH_POD:?BENCHSUITE_ATTACH_POD required}"
  cleanup_lium() { log "releasing Lium pod $POD"; "$PY" "$HERE/kingpod.py" release "$POD" || true; }
  trap cleanup_lium EXIT
  export LIUM_API_KEY="${LIUM_API_KEY:-${LIUM:-}}"
  local HOST PORT API_KEY
  HOST=$("$PY" -c 'import json; print(json.load(open("'"$HERE"'/state/pods.json"))["'"$POD"'"]["ssh_host"])')
  PORT=$("$PY" -c 'import json; print(json.load(open("'"$HERE"'/state/pods.json"))["'"$POD"'"]["ssh_port"])')
  API_KEY=$("$PY" -c 'import json; print(json.load(open("'"$HERE"'/state/pods.json"))["'"$POD"'"]["key"])')
  local USER_HOST="root@$HOST" SSH_KEY="$HOME/.ssh/id_ed25519" KH="$HERE/state/known_hosts" RHOME=/root
  SSH=(ssh -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=20 -o LogLevel=ERROR -p "$PORT" "$USER_HOST")
  local REMOTE_ENV="export BENCH_API_KEY='$API_KEY' PRIME_API_KEY='${PRIME_API_KEY:-}' HF_TOKEN='${HF_TOKEN:-}' BENCHSUITE_CHAT_IMAGE=affine-bench-chat:py311; cd $RHOME/affine/ops/benchsuite"
  local PYR="$RHOME/benchsuite/verifiers/.venv/bin/python"
  log "attached to $POD ($HOST:$PORT) for $RUN_ID; waiting for the pod's run_suite processes"
  while :; do
    local N; N=$("${SSH[@]}" "pgrep -fc '[r]un_suite.py run --run-id $RUN_ID' || true" 2>/dev/null || echo "ssh")
    [ "$N" = "ssh" ] && { log "ssh to the pod failed; retrying in 5 min"; sleep 300; continue; }
    [ "${N:-0}" -eq 0 ] && break
    pull_run "$USER_HOST" "$PORT" "$SSH_KEY" "$KH" "$RHOME"
    "$PY" "$HERE/publish.py" --run-dir "$RUN_DIR" --only-state --partial || true
    sleep 900
  done
  log "pod's cells finished; running the tail (retry, summarize, push, pull, publish)"
  "${SSH[@]}" "$REMOTE_ENV && $PYR run_suite.py retry --run-id $RUN_ID --out $RHOME/benchsuite/runs --verifiers-dir $RHOME/benchsuite/verifiers && $PYR run_suite.py summarize --run-id $RUN_ID --out $RHOME/benchsuite/runs" || log "retry/summarize returned non-zero"
  "${SSH[@]}" "$REMOTE_ENV && $PYR push_evals.py --run-dir $RHOME/benchsuite/runs/$RUN_ID --models king" || log "push_evals returned non-zero; continuing"
  pull_run "$USER_HOST" "$PORT" "$SSH_KEY" "$KH" "$RHOME"
  "$PY" "$HERE/publish.py" --run-dir "$RUN_DIR" || finish 8
  finish 0
}

pull_light() {  # user@host port key known_hosts remote_home — summaries + manifests only (per-cell publish)
  tar_cmd="cd $5/benchsuite/runs && tar czf - --exclude='*/logs' --exclude='*/traces.jsonl*' --exclude='*/eval.log' --exclude='*/harbor' $RUN_ID"
  ssh -i "$3" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$4" -o LogLevel=ERROR -o ConnectTimeout=20 -p "$2" "$1" "$tar_cmd" 2>/dev/null | tar xzf - -C "$BENCH_HOME/runs" 2>/dev/null
}

pull_run() {  # user@host port key known_hosts remote_home
  tar_cmd="cd $5/benchsuite/runs && tar czf - --exclude='*/logs/attempt_*' $RUN_ID"
  ssh -i "$3" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$4" -o LogLevel=ERROR -p "$2" "$1" "$tar_cmd" | tar xzf - -C "$BENCH_HOME/runs" || log "pull failed (will retry at the end)"
}

# ===================================================== comparables ==========
run_comparables() {
  local URL; URL=$(toml comparables.inference_url)
  local KEY_ENV; KEY_ENV=$(toml comparables.key_env)
  "$PY" - "$REF" "$LABEL" "$CODE_COMMIT" "$TEACHER_FROM" > "$RUN_DIR/meta.json" <<'PY'
import json, sys, tomllib
ref, label, commit, tfrom = sys.argv[1:]
price = next((m for m in tomllib.load(open("suite.toml", "rb"))["comparables"]["models"] if m["id"] == ref), {})
print(json.dumps({"mode": "comparables", "king": {"model": ref, "label": label, "usd_per_mtok": price},
                  "teacher": {"reused_from": tfrom},
                  "where": {"provider": "Prime Inference", "eval_driver": "ArbosLife box, docker runtime, chat sets only"},
                  "code": {"affine_commit": commit}}))
PY
  export BENCH_API_KEY="${!KEY_ENV}"
  cd "$HERE" && "$PY" $(suite_cmd "--king-url $URL --king-model $REF --models king --teacher-from $TEACHER_FROM --verifiers-dir $BENCH_HOME/verifiers --out $BENCH_HOME/runs" docker "$CHAT_ENVS" primary,secondary 32 manifest.json) --meta "$RUN_DIR/meta.json" || log "chat suite returned non-zero"
  "$PY" "$HERE/run_suite.py" retry --run-id "$RUN_ID" --out "$BENCH_HOME/runs" --verifiers-dir "$BENCH_HOME/verifiers" || true
  stamp_trigger never
  "$PY" "$HERE/publish.py" --run-dir "$RUN_DIR" || finish 8
  finish 0
}

# ========================================================== lium (default) ==
run_lium() {  # $1 = sandbox policy (gated|never)
  local POD=""
  cleanup_lium() { [ -n "$POD" ] && { log "releasing Lium pod $POD"; "$PY" "$HERE/kingpod.py" release "$POD" || true; }; }
  trap cleanup_lium EXIT
  export LIUM_API_KEY="${LIUM_API_KEY:-${LIUM:-}}"
  local DIGEST="$REF" R2FLAG=""
  if [[ "$REF" == r2://* ]]; then
    DIGEST="${CHALLENGER_REVISION:?CHALLENGER_REVISION (sha256 model_digest) required for an r2:// ref}"
    R2FLAG="--r2 $REF"
    export AFFINE_EVAL_R2_ENDPOINT="${AFFINE_EVAL_R2_ENDPOINT:-${R2_ENDPOINT:-}}"
  elif [[ "$REF" == hf://* ]]; then
    # genesis: hf://<repo>@<revision>; the card's digest is hf-<revision[:10]>
    local HFSPEC="${REF#hf://}"
    DIGEST="hf-$(echo "${HFSPEC#*@}" | cut -c1-10)"
    R2FLAG="--hf $HFSPEC"
  fi
  if [ -n "${BENCHSUITE_REUSE_POD:-}" ]; then
    POD="$BENCHSUITE_REUSE_POD"          # an already-serving pod (state ready in pods.json)
    log "reusing Lium pod $POD"
  else
    # shellcheck disable=SC2086
    POD=""
    local PLANS; PLANS="$(toml modes.lium_plan) $(toml modes.lium_plan_fallbacks | tr "," " ")"
    if [ "${BENCHSUITE_SANDBOX:-}" = "daytona" ] && [ "$(toml sandbox_daytona.concurrency)" -gt 50 ]; then
      # the sandboxes are never the limit; above ~50 in flight one replica is (2026-09-17: 64
      # running + 34 queued on a B200) -> two-replica pods first, so wall time is what the mode buys
      PLANS="$(toml sandbox_daytona.pod_plans | tr "," " ") $PLANS"
    fi
    for PLAN in $PLANS; do
      POD=$("$PY" "$HERE/kingpod.py" rent --plan "$PLAN" --digest "$DIGEST" $R2FLAG | tail -1) && [ -n "$POD" ] && break
      log "no pod on plan $PLAN; trying the next plan"; POD=""
    done
    [ -n "$POD" ] || finish 2
    "$PY" "$HERE/kingpod.py" wait "$POD" > /dev/null || finish 3
  fi
  local MEM; MEM=$("$PY" -c 'import json; m=json.load(open("'"$HERE"'/state/pods.json"))["'"$POD"'"]; print(json.dumps({k:m[k] for k in ("ssh_host","ssh_port","key","price","served","base_url")}))')
  local HOST PORT API_KEY USD_HR SERVED
  HOST=$(echo "$MEM" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["ssh_host"])')
  PORT=$(echo "$MEM" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["ssh_port"])')
  API_KEY=$(echo "$MEM" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["key"])')
  USD_HR=$(echo "$MEM" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["price"])')
  SERVED=$(echo "$MEM" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["served"])')
  local FRONT; FRONT=$("$PY" -c 'import json; print(json.load(open("'"$HERE"'/state/pods.json"))["'"$POD"'"]["front_internal"])' 2>/dev/null || echo "")
  # on the pod itself the king answers on the nginx front port (loopback); the driver runs there
  local KING_URL="http://127.0.0.1:${FRONT}/v1"
  log "Lium pod $POD ($USD_HR/h) serving $SERVED; driver on the pod"
  remote_suite king "$1" docker "root@$HOST" "$PORT" "$HOME/.ssh/id_ed25519" "$HERE/state/known_hosts" "$API_KEY" \
    "$USD_HR" "$POD" "$KING_URL" "$SERVED" "" "" "Lium (our fleet, TAO)"
  finish 0
}

log "pass $RUN_ID mode=$MODE ref=$REF label=$LABEL code=$CODE_COMMIT"
case "$MODE" in
  lium)        run_lium gated ;;
  prime)       run_on_prime_pod king gated ;;
  full)        run_on_prime_pod king,teacher always ;;
  challenger)  run_lium never ;;
  genesis)     run_lium never ;;
  agentic)     run_lium never ;;   # modes.agentic_envs on its own pod; BENCHSUITE_MERGE_INTO merges into a card
  attach)      attach_lium ;;   # hf://<repo>@<rev> ref, chat sets only, teacher reused; the kingboard's Genesis row
  comparables) run_comparables ;;
  *) log "unknown mode $MODE"; finish 9 ;;
esac
