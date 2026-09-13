#!/bin/bash
# One benchmark-suite pass, end to end, on Prime Intellect's stack.
#
#   run_pass.sh <model_ref> <label> <run_id> [MODE]        MODE defaults to [modes].default
#
#   prime        <model_ref> = king sha256 digest (public models.affine.io copy),
#                <label> = reign number. King only on a Prime pod (4 TP2 replicas),
#                teacher reused from [modes].teacher_from; chat sets always,
#                sandbox sets when a chat cell moved vs the previous published run
#                (or BENCHSUITE_FORCE_SANDBOX=1). The standing per-crown / weekly mode.
#   full         same pod, king + teacher served (2 TP2 each), every env, no gate.
#                The reference pass that refreshes the teacher baseline.
#   challenger   <model_ref> = r2://affine-private-models/models/registrations/<reg>/
#                (+ CHALLENGER_REVISION=<sha256>), <label> = chal-NNNNN. King-only pod,
#                chat sets only, teacher reused. Needs AFFINE_EVAL_R2_* (read-only key).
#   comparables  <model_ref> = Prime Inference model id (e.g. qwen/qwen3.6-35b-a3b),
#                <label> = display label. No pod: the eval driver runs here (docker),
#                the model is Prime Inference, chat sets only, teacher reused.
#   cheap        fallback: king on a Lium 1x H200 (kingpod.py), chat sets, teacher
#                reused; sandbox sets on Prime only when a chat cell moved.
#
# Env (ops/benchsuite/run.sh loads the box snapshot): PRIME_API_KEY (or PRIME),
# HF_TOKEN, DATA_R2_*, LIUM_API_KEY (cheap), AFFINE_EVAL_R2_* (challenger),
# PRIME_SSH_KEY (default ~/.ssh/prime_bench, registered on the Prime account).
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
log() { echo "[run_pass] $(date -u +%FT%TZ) $*"; }
finish() { echo "$1" > "$LOG_EXIT"; exit "$1"; }
CODE_COMMIT=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)
RUN_DIR="$BENCH_HOME/runs/$RUN_ID"
mkdir -p "$RUN_DIR"
export PRIME_API_KEY="${PRIME_API_KEY:-${PRIME:-}}"
TEACHER_FROM=$(toml modes.teacher_from)
CHAT_ENVS=$(toml modes.chat_envs)
SANDBOX_ENVS=$(toml modes.sandbox_envs)
PREV_CARD=$(ls -t "$REPO/$(toml suite.state_dir)"/*.json 2>/dev/null | head -1)

# The eval driver, run either here or on the pod (RUNNER="ssh ..." prefix).
# $1 = extra run_suite.py flags, $2 = runtime, $3 = envs, $4 = temps, $5 = concurrency, $6 = manifest
suite_cmd() {
  echo "run_suite.py run --run-id $RUN_ID --key-env BENCH_API_KEY --pod-usd-per-hour ${USD_HR:-0} --meta meta.json $1 --runtime $2 --envs $3 --temps $4 --concurrency $5 --parallel-envs 2 --manifest $6"
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

  # eval env + suite code + teacher baseline copy, while weights download
  tar -C "$REPO" -czf "/tmp/benchsuite-$RUN_ID.tgz" ops/benchsuite
  "${SCP[@]}" "/tmp/benchsuite-$RUN_ID.tgz" "$USER_HOST:/tmp/benchsuite.tgz" || finish 4
  rm -f "/tmp/benchsuite-$RUN_ID.tgz"
  "${SSH[@]}" 'mkdir -p ~/affine ~/benchsuite/runs && cd ~/affine && tar xzf /tmp/benchsuite.tgz && BENCH_HOME=$HOME/benchsuite bash ~/affine/ops/benchsuite/install_eval_env.sh > ~/install.log 2>&1; tail -1 ~/install.log' || finish 5
  "${SSH[@]}" 'docker pull -q python:3.11-slim >/dev/null 2>&1 || sudo docker pull -q python:3.11-slim >/dev/null'
  if [ "$MODELS" = "king" ] && [ -d "$BENCH_HOME/runs/$TEACHER_FROM" ]; then
    tar -C "$BENCH_HOME/runs" -czf "/tmp/teacher-$RUN_ID.tgz" --exclude 'traces.jsonl*' --exclude 'logs' "$TEACHER_FROM/teacher"
    "${SCP[@]}" "/tmp/teacher-$RUN_ID.tgz" "$USER_HOST:/tmp/teacher.tgz" && "${SSH[@]}" 'cd ~/benchsuite/runs && tar xzf /tmp/teacher.tgz'
    rm -f "/tmp/teacher-$RUN_ID.tgz"
  fi
  for _ in $(seq 1 120); do
    "${SSH[@]}" 'sudo test -f /root/bench/ready' 2>/dev/null && break
    if "${SSH[@]}" 'sudo test -f /root/bench/bootstrap.failed' 2>/dev/null; then log "bootstrap failed: $("${SSH[@]}" 'sudo cat /root/bench/bootstrap.failed')"; finish 6; fi
    sleep 30
  done
  "${SSH[@]}" 'sudo test -f /root/bench/ready' || { log "serving never became ready"; finish 6; }
  log "serving ready ($MODELS)"

  local META; META=$("$PY" - "$REF" "$LABEL" "$POD_ID" "$USD_HR" "$CODE_COMMIT" "$MODE" "$TEACHER_FROM" "$MODELS" <<'PY'
import json, sys
ref, label, pod, usd, commit, mode, tfrom, models = sys.argv[1:]
king = {"digest": ref} if not ref.startswith("r2://") else {"repo": ref}
if label.isdigit(): king["reign"] = int(label)
else: king["label"] = label
meta = {"mode": mode, "king": king,
        "where": {"provider": "Prime Intellect pod", "pod_id": pod, "usd_per_hour": float(usd),
                  "eval_driver": "same pod; docker runtime for chat sets, Prime sandboxes for sandbox sets"},
        "code": {"affine_commit": commit}}
if models == "king": meta["teacher"] = {"reused_from": tfrom}
print(json.dumps(meta))
PY
)
  local MODEL_FLAGS="--king-url http://127.0.0.1:8001/v1 --king-model king --models $MODELS --verifiers-dir ~/benchsuite/verifiers --out ~/benchsuite/runs --push"
  [ "$MODELS" = "king,teacher" ] && MODEL_FLAGS="$MODEL_FLAGS --teacher-url http://127.0.0.1:8002/v1 --teacher-model teacher" || MODEL_FLAGS="$MODEL_FLAGS --teacher-from $TEACHER_FROM"
  local REMOTE_ENV="export BENCH_API_KEY='$API_KEY' PRIME_API_KEY='${PRIME_API_KEY:-}' HF_TOKEN='${HF_TOKEN:-}'; cd ~/affine/ops/benchsuite && echo '$META' > meta.json"
  "${SSH[@]}" "$REMOTE_ENV && ~/benchsuite/verifiers/.venv/bin/python $(suite_cmd "$MODEL_FLAGS" docker "$CHAT_ENVS" primary,secondary 64 manifest.json)" || log "chat suite returned non-zero; continuing"
  # pull the chat results back now so the gate can compare against the published card
  pull_run "$USER_HOST" "$PORT" "$SSH_KEY" "$KH"
  local TRIGGER="always"
  if [ "$SANDBOX_POLICY" = "gated" ]; then TRIGGER=$(sandbox_trigger); fi
  if [ "$SANDBOX_POLICY" = "never" ]; then TRIGGER="never"; fi
  stamp_trigger "$TRIGGER"
  if [ "$TRIGGER" != "none" ] && [ "$TRIGGER" != "never" ]; then
    log "sandbox sets ($TRIGGER): SWE-bench Verified + miniF2F + long-context on Prime sandboxes"
    "${SSH[@]}" "$REMOTE_ENV && ~/benchsuite/verifiers/.venv/bin/python $(suite_cmd "$MODEL_FLAGS" prime "$SANDBOX_ENVS" primary 48 manifest-sandbox.json)" || log "sandbox suite returned non-zero; continuing"
  else
    log "no king chat cell moved beyond the previous run's interval; sandbox sets skipped"
  fi
  "${SSH[@]}" "$REMOTE_ENV && ~/benchsuite/verifiers/.venv/bin/python run_suite.py retry --run-id $RUN_ID --out ~/benchsuite/runs --verifiers-dir ~/benchsuite/verifiers && ~/benchsuite/verifiers/.venv/bin/python run_suite.py summarize --run-id $RUN_ID --out ~/benchsuite/runs" || log "retry/summarize returned non-zero"
  pull_run "$USER_HOST" "$PORT" "$SSH_KEY" "$KH"
  "$PY" "$HERE/publish.py" --run-dir "$RUN_DIR" || finish 8
  finish 0
}

pull_run() {  # user@host port key known_hosts
  tar_cmd="cd ~/benchsuite/runs && tar czf - --exclude='*/logs/attempt_*' $RUN_ID"
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

# ========================================================== cheap ===========
run_cheap() {
  local POD=""
  cleanup_cheap() { [ -n "$POD" ] && { log "releasing Lium pod $POD"; "$PY" "$HERE/kingpod.py" release "$POD" || true; }; }
  trap cleanup_cheap EXIT
  export LIUM_API_KEY="${LIUM_API_KEY:-${LIUM:-}}"
  POD=$("$PY" "$HERE/kingpod.py" rent --plan "$(toml modes.cheap_plan)" --digest "$REF" | tail -1) || finish 2
  local BASE; BASE=$("$PY" "$HERE/kingpod.py" wait "$POD" | tail -1) || finish 3
  export BENCH_API_KEY; BENCH_API_KEY=$("$PY" -c 'import json; print(json.load(open("'"$HERE"'/state/pods.json"))["'"$POD"'"]["key"])')
  USD_HR=$("$PY" -c 'import json; print(json.load(open("'"$HERE"'/state/pods.json"))["'"$POD"'"]["price"])')
  echo "{\"mode\": \"cheap\", \"king\": {\"digest\": \"$REF\", \"reign\": $LABEL}, \"teacher\": {\"reused_from\": \"$TEACHER_FROM\"}, \"where\": {\"provider\": \"Lium (our fleet)\", \"pod\": \"$POD\", \"usd_per_hour\": $USD_HR}, \"code\": {\"affine_commit\": \"$CODE_COMMIT\"}}" > "$RUN_DIR/meta.json"
  local MF="--king-url $BASE --king-model king-${REF:0:12} --models king --teacher-from $TEACHER_FROM --verifiers-dir $BENCH_HOME/verifiers --out $BENCH_HOME/runs"
  cd "$HERE" && "$PY" $(suite_cmd "$MF" docker "$CHAT_ENVS" primary,secondary 64 manifest.json) --meta "$RUN_DIR/meta.json" || log "chat suite returned non-zero"
  local TRIGGER; TRIGGER=$(sandbox_trigger); stamp_trigger "$TRIGGER"
  if [ "$TRIGGER" != "none" ]; then
    "$PY" $(suite_cmd "$MF" prime "$SANDBOX_ENVS" primary 32 manifest-sandbox.json) --meta "$RUN_DIR/meta.json" || log "sandbox suite returned non-zero"
  fi
  "$PY" "$HERE/publish.py" --run-dir "$RUN_DIR" || finish 8
  finish 0
}

log "pass $RUN_ID mode=$MODE ref=$REF label=$LABEL code=$CODE_COMMIT"
case "$MODE" in
  prime)       run_on_prime_pod king gated ;;
  full)        run_on_prime_pod king,teacher always ;;
  challenger)  run_on_prime_pod king never ;;
  comparables) run_comparables ;;
  cheap)       run_cheap ;;
  *) log "unknown mode $MODE"; finish 9 ;;
esac
