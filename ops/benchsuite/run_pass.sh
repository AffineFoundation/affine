#!/bin/bash
# One benchmark-suite pass for one king, end to end.
#
#   run_pass.sh <king_digest> <reign> <run_id> [cheap|full]      (default: [modes].default)
#
# cheap (the standing per-reign run; target < $50):
#   1. rent a Lium 1x H200 with kingpod.py, serve the king there (bootstrap_king.sh:
#      vLLM 0.28.0 + the chat-box parsers), wait for READY
#   2. run the chat sets for the KING ONLY from this box (docker runtime); the
#      teacher baseline is copied from [modes].cheap_teacher_from
#   3. run_suite.py compare against the previous published run of any reign:
#      if a king chat cell moved outside the previous 95% interval, or
#      BENCHSUITE_FORCE_SANDBOX=1, also run the sandbox sets (SWE-bench
#      Verified, miniF2F, long-context) on Prime sandboxes, king only
#   4. publish (R2 research/benchsuite/<run_id>/ + affine/state/benchsuite/) and
#      release the Lium pod (always, via trap)
#
# full (the reference pass; first run 2026-09-12 on reign 11):
#   king + teacher on one Prime pod ([prime] in suite.toml), every env; the
#   eval driver runs on the pod; results rsynced back and published; pod
#   terminated (always, via trap).
#
# Env: PRIME_API_KEY (sandboxes / Prime pod), HF_TOKEN (gated datasets),
# DATA_R2_* (publish), LIUM_API_KEY (cheap), PRIME_SSH_KEY (full). Loaded by
# ops/benchsuite/run.sh from the box env snapshot. Exit code -> <log>.exit for
# watch.py.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="${BENCHSUITE_PYTHON:-$REPO/.venv/bin/python}"
BENCH_HOME="${BENCH_HOME:-$HOME/benchsuite}"
DIGEST="$1"; REIGN="$2"; RUN_ID="$3"
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

# ============================================================ cheap ==========
run_cheap() {
  local POD=""
  cleanup_cheap() { [ -n "$POD" ] && { log "releasing Lium pod $POD"; "$PY" "$HERE/kingpod.py" release "$POD" || true; }; }
  trap cleanup_cheap EXIT
  export LIUM_API_KEY="${LIUM_API_KEY:-${LIUM:-}}"
  POD=$("$PY" "$HERE/kingpod.py" rent --plan "$(toml modes.cheap_plan)" --digest "$DIGEST" | tail -1) || finish 2
  local BASE; BASE=$("$PY" "$HERE/kingpod.py" wait "$POD" | tail -1) || finish 3
  local EP; EP=$("$PY" "$HERE/kingpod.py" endpoint "$POD")
  local MODEL; MODEL=$(echo "$EP" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["model"])')
  export BENCH_KING_KEY; BENCH_KING_KEY=$("$PY" -c 'import json; print(json.load(open("'"$HERE"'/state/pods.json"))["'"$POD"'"]["key"])')
  local PRICE; PRICE=$("$PY" -c 'import json; print(json.load(open("'"$HERE"'/state/pods.json"))["'"$POD"'"]["price"])')
  log "king served at $BASE as $MODEL (\$$PRICE/h)"
  local META; META=$("$PY" - "$DIGEST" "$REIGN" "$POD" "$PRICE" "$CODE_COMMIT" "$(toml modes.cheap_teacher_from)" <<'PY'
import json, sys
digest, reign, pod, price, commit, tfrom = sys.argv[1:]
print(json.dumps({"mode": "cheap", "king": {"digest": digest, "reign": int(reign)},
                  "teacher": {"reused_from": tfrom},
                  "where": {"provider": "Lium (our fleet)", "pod": pod, "usd_per_hour": float(price),
                            "eval_driver": "ArbosLife box, docker runtime; sandbox sets on Prime sandboxes when triggered"},
                  "code": {"affine_commit": commit}}))
PY
)
  echo "$META" > "$RUN_DIR/meta.json"
  local COMMON="--run-id $RUN_ID --verifiers-dir $BENCH_HOME/verifiers --out $BENCH_HOME/runs --king-url $BASE --king-model $MODEL --key-env BENCH_KING_KEY --models king --pod-usd-per-hour $PRICE --meta $RUN_DIR/meta.json"
  # shellcheck disable=SC2086
  "$PY" "$HERE/run_suite.py" run $COMMON --runtime docker --envs "$(toml modes.cheap_chat_envs)" \
      --temps primary,secondary --concurrency 64 --parallel-envs 2 --teacher-from "$(toml modes.cheap_teacher_from)" \
      || log "chat suite returned non-zero; continuing"
  # sandbox trigger: movement vs the newest published scorecard, or the flag
  local PREV; PREV=$(ls -t "$REPO/$(toml suite.state_dir)"/*.json 2>/dev/null | head -1)
  local TRIGGER="none"
  if [ "${BENCHSUITE_FORCE_SANDBOX:-0}" = "1" ]; then TRIGGER="flag"
  elif [ -n "$PREV" ] && "$PY" "$HERE/run_suite.py" compare --run-id "$RUN_ID" --out "$BENCH_HOME/runs" --against "$PREV" | tee "$RUN_DIR/compare.txt" | grep -q MOVED; then TRIGGER="moved"
  fi
  "$PY" - "$RUN_DIR/manifest.json" "$TRIGGER" "$PREV" <<'PY'
import json, sys
p, trig, prev = sys.argv[1:]
m = json.load(open(p)); m["sandbox_trigger"] = {"trigger": trig, "compared_against": prev}; json.dump(m, open(p, "w"), indent=1)
PY
  if [ "$TRIGGER" != "none" ]; then
    log "sandbox sets triggered ($TRIGGER): SWE-bench Verified + miniF2F + long-context on Prime sandboxes, king only"
    export PRIME_API_KEY="${PRIME_API_KEY:-${PRIME:-}}"
    # shellcheck disable=SC2086
    "$PY" "$HERE/run_suite.py" run $COMMON --runtime prime --envs "$(toml modes.cheap_sandbox_envs)" \
        --temps primary --concurrency 32 --parallel-envs 2 --manifest manifest-sandbox.json \
        --teacher-from "$(toml modes.cheap_teacher_from)" || log "sandbox suite returned non-zero; continuing"
  else
    log "no king chat cell moved beyond the previous run's interval; sandbox sets skipped"
  fi
  "$PY" "$HERE/publish.py" --run-dir "$RUN_DIR" || finish 8
  finish 0
}

# ============================================================= full ==========
run_full() {
  local POD_ID=""
  local SSH_KEY="${PRIME_SSH_KEY:-$HOME/.ssh/prime_bench}"
  cleanup_full() { [ -n "$POD_ID" ] && { log "terminating Prime pod $POD_ID"; "$PY" "$HERE/primepod.py" terminate "$POD_ID" || true; }; }
  trap cleanup_full EXIT
  export PRIME_API_KEY="${PRIME_API_KEY:-${PRIME:-}}"
  local CREATE; CREATE=$("$PY" "$HERE/primepod.py" create --name "affine-benchsuite-$RUN_ID" | tail -1) || finish 2
  POD_ID=$(echo "$CREATE" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["pod_id"])')
  local USD_HR; USD_HR=$(echo "$CREATE" | "$PY" -c 'import json,sys; print(json.load(sys.stdin).get("usd_per_hour") or 0)')
  local SSH_TARGET; SSH_TARGET=$("$PY" "$HERE/primepod.py" wait "$POD_ID" --timeout-min 40) || finish 3
  local USER_HOST=${SSH_TARGET% *} PORT=${SSH_TARGET##* }
  local KH="$HERE/state/prime_known_hosts"
  SSH=(ssh -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=20 -o LogLevel=ERROR -p "$PORT" "$USER_HOST")
  SCP=(scp -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o LogLevel=ERROR -P "$PORT")
  log "pod $POD_ID up at $USER_HOST:$PORT (\$$USD_HR/h)"

  local API_KEY; API_KEY=$("$PY" -c 'import secrets; print(secrets.token_hex(24))')
  local ENV_TMP; ENV_TMP=$(mktemp /tmp/benchenv.XXXXXX)
  cat > "$ENV_TMP" <<EOF
KING_DIGEST="$DIGEST"
TEACHER_HF="$(toml teacher.hf_repo)"
HF_TOKEN="${HF_TOKEN:-}"
KING_REPLICAS="$(toml prime.king_replicas)"
TEACHER_REPLICAS="$(toml prime.teacher_replicas)"
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

  tar -C "$REPO" -czf "/tmp/benchsuite-$RUN_ID.tgz" ops/benchsuite
  "${SCP[@]}" "/tmp/benchsuite-$RUN_ID.tgz" "$USER_HOST:/tmp/benchsuite.tgz" || finish 4
  rm -f "/tmp/benchsuite-$RUN_ID.tgz"
  "${SSH[@]}" 'mkdir -p ~/affine && cd ~/affine && tar xzf /tmp/benchsuite.tgz && BENCH_HOME=$HOME/benchsuite bash ~/affine/ops/benchsuite/install_eval_env.sh > ~/install.log 2>&1; tail -1 ~/install.log' || finish 5
  "${SSH[@]}" 'docker pull -q python:3.11-slim >/dev/null 2>&1 || sudo docker pull -q python:3.11-slim >/dev/null'
  for _ in $(seq 1 120); do
    "${SSH[@]}" 'sudo test -f /root/bench/ready' 2>/dev/null && break
    if "${SSH[@]}" 'sudo test -f /root/bench/bootstrap.failed' 2>/dev/null; then log "bootstrap failed: $("${SSH[@]}" 'sudo cat /root/bench/bootstrap.failed')"; finish 6; fi
    sleep 30
  done
  "${SSH[@]}" 'sudo test -f /root/bench/ready' || { log "serving never became ready"; finish 6; }
  log "king + teacher serving"

  local META; META=$("$PY" - "$DIGEST" "$REIGN" "$POD_ID" "$USD_HR" "$CODE_COMMIT" <<'PY'
import json, sys
digest, reign, pod, usd, commit = sys.argv[1:]
print(json.dumps({"mode": "full", "king": {"digest": digest, "reign": int(reign)},
                  "where": {"provider": "Prime Intellect pod", "pod_id": pod, "usd_per_hour": float(usd),
                            "eval_driver": "same pod; docker runtime for chat sets, Prime sandboxes for sandbox sets"},
                  "code": {"affine_commit": commit}}))
PY
)
  local CHAT; CHAT=$(toml modes.cheap_chat_envs)
  local SANDBOX; SANDBOX=$(toml modes.cheap_sandbox_envs)
  local COMMON="--run-id $RUN_ID --verifiers-dir ~/benchsuite/verifiers --out ~/benchsuite/runs --king-url http://127.0.0.1:8001/v1 --king-model king --teacher-url http://127.0.0.1:8002/v1 --teacher-model teacher --key-env BENCH_API_KEY --models king,teacher --pod-usd-per-hour $USD_HR --push"
  "${SSH[@]}" "export BENCH_API_KEY='$API_KEY' PRIME_API_KEY='${PRIME_API_KEY:-}' HF_TOKEN='${HF_TOKEN:-}'; cd ~/affine/ops/benchsuite && echo '$META' > meta.json && \
    ~/benchsuite/verifiers/.venv/bin/python run_suite.py run $COMMON --runtime docker --envs $CHAT --temps primary,secondary --concurrency 64 --parallel-envs 2 --meta meta.json; \
    ~/benchsuite/verifiers/.venv/bin/python run_suite.py run $COMMON --runtime prime --envs $SANDBOX --temps primary --concurrency 32 --parallel-envs 2 --manifest manifest-sandbox.json --meta meta.json; \
    ~/benchsuite/verifiers/.venv/bin/python run_suite.py retry --run-id $RUN_ID --out ~/benchsuite/runs --verifiers-dir ~/benchsuite/verifiers" || log "suite returned non-zero; publishing what finished"
  rsync -a -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$KH -p $PORT" \
    --exclude 'logs/attempt_*' "$USER_HOST:benchsuite/runs/$RUN_ID/" "$RUN_DIR/" || finish 7
  "$PY" "$HERE/publish.py" --run-dir "$RUN_DIR" || finish 8
  finish 0
}

log "pass $RUN_ID mode=$MODE king=$DIGEST reign=$REIGN code=$CODE_COMMIT"
case "$MODE" in
  cheap) run_cheap ;;
  full) run_full ;;
  *) log "unknown mode $MODE"; finish 9 ;;
esac
