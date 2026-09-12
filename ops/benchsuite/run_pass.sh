#!/bin/bash
# One full benchmark-suite pass for one king, end to end on a Prime Intellect
# pod (the path used for the first reign-11 run, 2026-09-12):
#
#   run_pass.sh <king_digest> <reign> <run_id>
#
#   1. create a Prime pod per [prime] in suite.toml; wait for ssh
#   2. push /root/bench/env + pod_bootstrap.sh -> vLLM 0.28 serves king (8001) + teacher (8002)
#   3. install the eval driver env on the pod (install_eval_env.sh) and the suite code
#   4. run every env: chat sets under the docker runtime, sandbox sets on Prime sandboxes
#   5. rsync the run directory back to $BENCH_HOME/runs/<run_id> on this box
#   6. publish.py -> R2 research/benchsuite/<run_id>/ + affine/state/benchsuite/<run_id>.json
#   7. terminate the pod (always, via trap)
#
# Env (from ~/.affine-validator.env / repo .env, loaded by ops/benchsuite/run.sh):
#   PRIME_API_KEY, HF_TOKEN, DATA_R2_*; PRIME_SSH_KEY (default ~/.ssh/prime_bench).
# Exit code is written next to the pass log as <log>.exit for watch.py.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="${BENCHSUITE_PYTHON:-$REPO/.venv/bin/python}"
BENCH_HOME="${BENCH_HOME:-$HOME/benchsuite}"
DIGEST="$1"; REIGN="$2"; RUN_ID="$3"
SSH_KEY="${PRIME_SSH_KEY:-$HOME/.ssh/prime_bench}"
LOG_EXIT="$HERE/state/pass-$RUN_ID.exit"
log() { echo "[run_pass] $(date -u +%FT%TZ) $*"; }
finish() { echo "$1" > "$LOG_EXIT"; exit "$1"; }

POD_ID=""
cleanup() {
  if [ -n "$POD_ID" ]; then
    log "terminating pod $POD_ID"
    "$PY" "$HERE/primepod.py" terminate "$POD_ID" || true
  fi
}
trap cleanup EXIT

# 1. pod
CREATE=$("$PY" "$HERE/primepod.py" create --name "affine-benchsuite-$RUN_ID" | tail -1) || finish 2
POD_ID=$(echo "$CREATE" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["pod_id"])')
USD_HR=$(echo "$CREATE" | "$PY" -c 'import json,sys; print(json.load(sys.stdin).get("usd_per_hour") or 0)')
SSH_TARGET=$("$PY" "$HERE/primepod.py" wait "$POD_ID" --timeout-min 40) || finish 3
USER_HOST=${SSH_TARGET% *}; PORT=${SSH_TARGET##* }
SSH=(ssh -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$HERE/state/prime_known_hosts" -o ConnectTimeout=20 -o LogLevel=ERROR -p "$PORT" "$USER_HOST")
SCP=(scp -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$HERE/state/prime_known_hosts" -o LogLevel=ERROR -P "$PORT")
log "pod $POD_ID up at $USER_HOST:$PORT (\$$USD_HR/h)"

# 2. serving
API_KEY=$("$PY" -c 'import secrets; print(secrets.token_hex(24))')
TEACHER_HF=$("$PY" -c 'import tomllib; print(tomllib.load(open("'"$HERE"'/suite.toml","rb"))["teacher"]["hf_repo"])')
read_prime() { "$PY" -c 'import tomllib,sys; print(tomllib.load(open("'"$HERE"'/suite.toml","rb"))["prime"][sys.argv[1]])' "$1"; }
read_serving() { "$PY" -c 'import tomllib,sys; print(tomllib.load(open("'"$HERE"'/suite.toml","rb"))["serving"][sys.argv[1]])' "$1"; }
ENV_TMP=$(mktemp)
cat > "$ENV_TMP" <<EOF
KING_DIGEST="$DIGEST"
TEACHER_HF="$TEACHER_HF"
HF_TOKEN="${HF_TOKEN:-}"
KING_REPLICAS="$(read_prime king_replicas)"
TEACHER_REPLICAS="$(read_prime teacher_replicas)"
API_KEY="$API_KEY"
MAX_MODEL_LEN="$(read_serving max_model_len)"
GPU_UTIL="$(read_serving gpu_memory_utilization)"
BATCHED_TOKENS="$(read_serving max_num_batched_tokens)"
MAX_NUM_SEQS="$(read_serving max_num_seqs)"
VLLM_VERSION="$(read_serving vllm_version)"
VLLM_CUDA="${VLLM_CUDA:-cu129}"
EOF
"${SCP[@]}" "$ENV_TMP" "$HERE/pod_bootstrap.sh" "$USER_HOST:/tmp/" || finish 4
rm -f "$ENV_TMP"
"${SSH[@]}" 'sudo mkdir -p /root/bench /root/logs && sudo mv /tmp/'"$(basename "$ENV_TMP")"' /root/bench/env && sudo chmod 600 /root/bench/env && sudo mv /tmp/pod_bootstrap.sh /root/bench/bootstrap.sh && sudo bash -c "cd /root && HOME=/root nohup setsid bash /root/bench/bootstrap.sh >> /root/bench/bootstrap.log 2>&1 < /dev/null &" ; sudo usermod -aG docker $USER 2>/dev/null; echo launched' || finish 4

# 3. eval env + suite code (in parallel with the model download)
tar -C "$REPO" -czf /tmp/benchsuite-$RUN_ID.tgz ops/benchsuite
"${SCP[@]}" /tmp/benchsuite-$RUN_ID.tgz "$USER_HOST:/tmp/benchsuite.tgz" || finish 4
rm -f /tmp/benchsuite-$RUN_ID.tgz
"${SSH[@]}" 'mkdir -p ~/affine && cd ~/affine && tar xzf /tmp/benchsuite.tgz && BENCH_HOME=$HOME/benchsuite bash ~/affine/ops/benchsuite/install_eval_env.sh > ~/install.log 2>&1; tail -1 ~/install.log' || finish 5
"${SSH[@]}" 'docker pull -q python:3.11-slim >/dev/null 2>&1 || sudo docker pull -q python:3.11-slim >/dev/null'

# wait for READY (both models)
for _ in $(seq 1 120); do
  if "${SSH[@]}" 'sudo test -f /root/bench/ready' 2>/dev/null; then break; fi
  if "${SSH[@]}" 'sudo test -f /root/bench/bootstrap.failed' 2>/dev/null; then log "bootstrap failed: $("${SSH[@]}" 'sudo cat /root/bench/bootstrap.failed')"; finish 6; fi
  sleep 30
done
"${SSH[@]}" 'sudo test -f /root/bench/ready' || { log "serving never became ready"; finish 6; }
log "king + teacher serving"

# 4. run (chat sets on docker, sandbox sets on Prime sandboxes)
META=$("$PY" - "$DIGEST" "$REIGN" "$POD_ID" "$USD_HR" "$RUN_ID" <<'PY'
import json, subprocess, sys
digest, reign, pod, usd, run_id = sys.argv[1:]
commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
print(json.dumps({"king": {"digest": digest, "reign": int(reign)},
                  "where": {"provider": "Prime Intellect pod", "pod_id": pod, "usd_per_hour": float(usd),
                            "eval_driver": "same pod; docker runtime for chat sets, Prime sandboxes for sandbox sets"},
                  "code": {"affine_commit": commit}}))
PY
)
CHAT_ENVS=$("$PY" -c 'import tomllib; d=tomllib.load(open("'"$HERE"'/suite.toml","rb")); print(",".join(e["id"] for e in d["envs"] if e["runtime"]=="none" or e["id"] in ("humaneval","livecodebench","bfcl-v3")))')
SANDBOX_ENVS=$("$PY" -c 'import tomllib; d=tomllib.load(open("'"$HERE"'/suite.toml","rb")); print(",".join(e["id"] for e in d["envs"] if e["runtime"]=="sandbox" and e["id"] not in ("humaneval","livecodebench","bfcl-v3")))')
RUN_COMMON="--run-id $RUN_ID --verifiers-dir ~/benchsuite/verifiers --out ~/benchsuite/runs --king-url http://127.0.0.1:8001/v1 --king-model king --teacher-url http://127.0.0.1:8002/v1 --teacher-model teacher --key-env BENCH_API_KEY --models king,teacher --pod-usd-per-hour $USD_HR --push"
"${SSH[@]}" "export BENCH_API_KEY='$API_KEY' PRIME_API_KEY='${PRIME_API_KEY:-}' HF_TOKEN='${HF_TOKEN:-}'; cd ~/affine/ops/benchsuite && echo '$META' > meta.json && \
  ~/benchsuite/verifiers/.venv/bin/python run_suite.py run $RUN_COMMON --runtime docker --envs $CHAT_ENVS --temps primary,secondary --concurrency 64 --meta meta.json && \
  ~/benchsuite/verifiers/.venv/bin/python run_suite.py run $RUN_COMMON --runtime prime --envs $SANDBOX_ENVS --temps primary --concurrency 32 --meta meta.json" || log "suite returned non-zero; publishing what finished"

# 5. pull back + 6. publish
mkdir -p "$BENCH_HOME/runs"
rsync -a -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$HERE/state/prime_known_hosts -p $PORT" \
  --exclude 'logs/attempt_*' "$USER_HOST:benchsuite/runs/$RUN_ID/" "$BENCH_HOME/runs/$RUN_ID/" || finish 7
"$PY" "$HERE/publish.py" --run-dir "$BENCH_HOME/runs/$RUN_ID" || finish 8
log "pass $RUN_ID published"
finish 0
