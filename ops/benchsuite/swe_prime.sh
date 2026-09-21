#!/bin/bash
# SWE-bench Verified @4h250 through Harbor's local docker runtime on a PRIME CPU node (real dockerd,
# address pools set at boot by primepod.py docker_bootstrap), against a Lium model box that is
# already serving. The parallel lane to Daytona (Jacob 2026-09-21 03:32 UTC): whichever lane lands
# the cell first publishes; this lane stops itself when the cell appears from elsewhere.
#   swe_prime.sh <digest> <label> <run_id> <model_pod> [in_flight] [plans]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"; PY="${BENCHSUITE_PYTHON:-$REPO/.venv/bin/python}"; BENCH_HOME="${BENCH_HOME:-$HOME/benchsuite}"
source "$HERE/env.sh"
export PRIME_API_KEY="${PRIME_API_KEY:-${PRIME:-}}"
DIGEST="$1" LABEL="$2" RUN_ID="$3" MODELPOD="$4" INFLIGHT="${5:-100}" PLANS="${6:-prime-cpu96 prime-cpu64 prime-cpu48}"
TAG=4h250; INTO="$BENCH_HOME/runs/$RUN_ID"; CELL="$INTO/king/swebench-verified@${TAG}__t0"
log() { echo "[swe-prime $LABEL] $(date -u +%FT%TZ) $*"; }
echo $$ > "$HERE/state/swe-local-${DIGEST:0:12}.pid"        # rows_watch: "swe4h(local docker lane)"
podf() { "$PY" -c 'import json,sys; m=json.load(open("'"$HERE"'/state/pods.json"))[sys.argv[1]]; print(m.get(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else ""))' "$1" "$2" "${3:-}"; }
MODEL_URL=$(podf "$MODELPOD" base_url); SERVED=$(podf "$MODELPOD" served); KEY=$(podf "$MODELPOD" key)
curl -s -m 15 -H "Authorization: Bearer $KEY" "$MODEL_URL/models" | grep -q "$SERVED" || { log "model box $MODELPOD does not answer at $MODEL_URL"; exit 2; }

# ---- rent the docker host (retry on no stock for up to 2 h)
POD=""; T0=$(date +%s)
while [ $(( $(date +%s) - T0 )) -lt 7200 ]; do
  for plan in $PLANS; do POD=$("$PY" "$HERE/primepod.py" rent --plan "$plan" --digest "$DIGEST" 2>>"$INTO/swe-prime-rent.log" | tail -1) && [ -n "$POD" ] && break; POD=""; done
  [ -n "$POD" ] && break; log "no Prime CPU stock; retry in 3 min"; sleep 180
done
[ -n "$POD" ] || { log "no docker host within 2 h"; rm -f "$HERE/state/swe-local-${DIGEST:0:12}.pid"; exit 2; }
cleanup() { log "releasing $POD"; "$PY" "$HERE/primepod.py" release "$POD" >/dev/null 2>&1 || true; rm -f "$HERE/state/swe-local-${DIGEST:0:12}.pid"; }
trap cleanup EXIT
timeout 1800 "$PY" "$HERE/primepod.py" wait "$POD" >>"$INTO/swe-prime-rent.log" 2>&1 || { log "docker host never became ready"; exit 3; }
U=$(podf "$POD" ssh_user root); H=$(podf "$POD" ssh_host); P=$(podf "$POD" ssh_port 22)
SSHO=(-i "${PRIME_SSH_KEY:-$HOME/.ssh/prime_bench}" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$HERE/state/prime_known_hosts" -o ConnectTimeout=20 -o LogLevel=ERROR)
ssh_h() { local cmd="$*"; [ "$U" != root ] && cmd="sudo -n bash -c $(printf '%q' "$cmd")"; ssh "${SSHO[@]}" -p "$P" "$U@$H" "$cmd"; }
DOCKERHUB_OP_ITEM="${DOCKERHUB_OP_ITEM:-op://Arbos/e7y3qzb2flaczam4rindajho4m}"
DH_USER=$(op read --no-newline "$DOCKERHUB_OP_ITEM/username" 2>/dev/null || echo ""); DH_TOKEN=$(op read --no-newline "$DOCKERHUB_OP_ITEM/credential" 2>/dev/null || echo "")
[ -n "$DH_TOKEN" ] && printf '%s' "$DH_TOKEN" | ssh_h "docker login -u '$DH_USER' --password-stdin >/dev/null 2>&1" || true
CORES=$(ssh_h nproc); log "docker host $POD ($CORES vCPU) -> model $SERVED at $MODEL_URL, $INFLIGHT in flight"

# ---- the job (same knobs as harbor_cell.py: mini-swe-agent, 4 h / 250 steps, T 0, max_tokens 8192)
JOB=/root/harbor_jobs; NAME="swe4h-${DIGEST:0:12}"
ssh_h "mkdir -p $JOB"
echo '{"agent": {"step_limit": 250}, "model": {"model_kwargs": {"temperature": 0.0}}}' | ssh_h "cat > $JOB/mswea-$NAME.json"
HCMD="/root/harborenv/bin/harbor run -d swebench-verified@1.0 -a mini-swe-agent -e docker -m openai/$SERVED -n $INFLIGHT -y -q -o $JOB --job-name $NAME --ae OPENAI_API_BASE=$MODEL_URL --ae OPENAI_BASE_URL=$MODEL_URL --ae OPENAI_API_KEY=$KEY --ae MSWEA_API_KEY=$KEY --agent-timeout-multiplier 4.8000 --max-retries 1 --ak max_tokens=8192 --ak config_file=$JOB/mswea-$NAME.json"
echo "$HCMD >> $JOB/$NAME.log 2>&1; echo HARBOR_EXIT=\$? >> $JOB/$NAME.log" | ssh_h "cat > $JOB/run-$NAME.sh"
ssh_h "tmux new-session -d -s $NAME 'bash $JOB/run-$NAME.sh'; tmux new-session -d -s prune-$NAME 'while sleep 1200; do docker image prune -af >/dev/null 2>&1; docker network prune -f >/dev/null 2>&1; done'" || { log "could not start harbor"; exit 6; }
log "harbor started on $POD (tmux $NAME)"
mkdir -p "$CELL"; [ -f "$CELL/summary.json" ] || echo "swe_prime.sh: harbor docker runtime on Prime node $POD, model $SERVED, started $(date -u +%FT%TZ)" > "$CELL/cmd.txt"

pull() { rsync -az --rsync-path="sudo rsync" -e "ssh ${SSHO[*]} -p $P" --exclude 'agent/*.jsonl' "$U@$H:$JOB/$NAME/" "$CELL/harbor/" 2>/dev/null; }
summarize() {
  export BENCH_API_KEY="$KEY"
  "$PY" "$HERE/harbor_cell.py" resummarize --env swebench-verified --budget-tag $TAG --model "$SERVED" --model-label king --model-url "$MODEL_URL" --out "$INTO/king" --agent-timeout-s 14400 --step-limit 250 --concurrency "$INFLIGHT" >/dev/null 2>&1 && \
  "$PY" - "$CELL/summary.json" "$POD" <<'PY'
import json, sys; p, pod = sys.argv[1:]; s = json.load(open(p)); s["sandbox"] = "prime-docker"; s["runtime"] = "harbor/docker (Prime CPU node)"
s["harness_note"] = (s.get("harness_note") or "") + f"; local docker runtime on Prime node {pod} (Daytona lane full)"; json.dump(s, open(p, "w"), indent=1)
PY
}
other_lane_done() {  # the Daytona lane published this cell first (a summary that is not ours)
  [ -f "$CELL/summary.json" ] && ! grep -q "prime-docker" "$CELL/summary.json" && grep -q '"n": 500' "$CELL/summary.json"
}
T0=$(date +%s)
while :; do
  sleep 600
  if other_lane_done; then log "the Daytona lane published this cell first; stopping"; ssh_h "tmux kill-session -t $NAME 2>/dev/null; docker rm -f \$(docker ps -aq) >/dev/null 2>&1"; exit 0; fi
  pull; N=$(ls "$CELL"/harbor/*/result.json 2>/dev/null | wc -l)
  summarize; log "$N/500 trials after $(( ($(date +%s) - T0) / 60 )) min"
  "$PY" "$HERE/publish.py" --run-dir "$INTO" --only-state --partial >/dev/null 2>&1 || true
  ssh_h "grep -q HARBOR_EXIT $JOB/$NAME.log 2>/dev/null" && break
done
pull; summarize
"$PY" "$HERE/publish.py" --run-dir "$INTO" --only-cells "king/swebench-verified@${TAG}__t0" 2>&1 | tail -1
log "done"
