#!/bin/bash
# SWE-bench Verified @4h250 through Harbor's LOCAL docker runtime on a Lium box (the pre-Daytona lane),
# against a model box that is already serving — a parallel lane when Daytona's slots are full
# (Jacob 2026-09-21 03:11 UTC). Harbor is installed on the docker host in its own venv; the job runs in
# tmux there; the job dir is pulled every 10 min, summarised with harbor_cell.py resummarize and
# published as swebench-verified@4h250.
#   swe_local.sh <digest> <label> <run_id> <docker_host_pod> [model_pod] [in_flight]
#     docker_host_pod  a pods.json pod with DinD (the model's own env box when it has spare cores)
#     model_pod        the serving pod (default: the docker host itself, model reached on 127.0.0.1)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"; PY="${BENCHSUITE_PYTHON:-$REPO/.venv/bin/python}"; BENCH_HOME="${BENCH_HOME:-$HOME/benchsuite}"
source "$HERE/env.sh"
DIGEST="$1" LABEL="$2" RUN_ID="$3" HOSTPOD="$4" MODELPOD="${5:-$4}" INFLIGHT="${6:-50}"
TAG=4h250; INTO="$BENCH_HOME/runs/$RUN_ID"; CELL="$INTO/king/swebench-verified@${TAG}__t0"
log() { echo "[swe-local $LABEL] $(date -u +%FT%TZ) $*"; }
echo $$ > "$HERE/state/swe-local-${DIGEST:0:12}.pid"
podf() { "$PY" -c 'import json,sys; m=json.load(open("'"$HERE"'/state/pods.json"))[sys.argv[1]]; print(m.get(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else ""))' "$1" "$2" "${3:-}"; }
SSHO=(-i "$HOME/.ssh/id_ed25519" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$HERE/state/known_hosts" -o ConnectTimeout=20 -o LogLevel=ERROR)
H_HOST=$(podf "$HOSTPOD" ssh_host); H_PORT=$(podf "$HOSTPOD" ssh_port)
ssh_h() { ssh "${SSHO[@]}" -p "$H_PORT" "root@$H_HOST" "$@"; }
if [ "$MODELPOD" = "$HOSTPOD" ]; then MODEL_URL="http://127.0.0.1:$(podf "$HOSTPOD" front_internal)/v1"; else MODEL_URL="$(podf "$MODELPOD" base_url)"; fi
SERVED=$(podf "$MODELPOD" served); KEY=$(podf "$MODELPOD" key)
DOCKERHUB_OP_ITEM="${DOCKERHUB_OP_ITEM:-op://Arbos/e7y3qzb2flaczam4rindajho4m}"
DH_USER=$(op read --no-newline "$DOCKERHUB_OP_ITEM/username" 2>/dev/null || echo ""); DH_TOKEN=$(op read --no-newline "$DOCKERHUB_OP_ITEM/credential" 2>/dev/null || echo "")

# ---- docker preflight (DinD overlay: the 09-17 genesis pod failed every `docker run` on overlay mounts)
if ! ssh_h "docker info >/dev/null 2>&1 && docker run --rm hello-world >/dev/null 2>&1 && echo DOCKER_OK" | grep -q DOCKER_OK; then
  log "docker preflight FAILED on $HOSTPOD (DinD/overlay); blacklisting its executor"; "$PY" "$HERE/kingpod.py" release "$HOSTPOD" --strike "docker preflight failed (swe_local)" >/dev/null 2>&1 || true
  exit 4
fi
[ -n "$DH_TOKEN" ] && printf '%s' "$DH_TOKEN" | ssh_h "docker login -u '$DH_USER' --password-stdin >/dev/null 2>&1" || true
# one compose network per trial: the default pool runs out ("all predefined address pools have been fully
# subnetted") after ~30 concurrent networks -> a /16 cut into /24s (250 networks) + prune leftovers
ssh_h 'mkdir -p /etc/docker; python3 - <<PY
import json, os
p = "/etc/docker/daemon.json"; d = json.load(open(p)) if os.path.exists(p) else {}
if not d.get("default-address-pools"):
    d["default-address-pools"] = [{"base": "10.200.0.0/16", "size": 24}, {"base": "10.201.0.0/16", "size": 24}]
    json.dump(d, open(p, "w")); print("pool set")
else: print("pool ok")
PY
docker network prune -f >/dev/null 2>&1
if ! docker network inspect bridge -f "{{.IPAM.Config}}" | grep -q 10.200; then (service docker restart >/dev/null 2>&1 || (pkill dockerd; sleep 3; nohup dockerd >/var/log/dockerd.log 2>&1 &)); sleep 8; fi
docker info >/dev/null 2>&1 && echo DOCKER_READY' | tail -1
CORES=$(ssh_h "nproc"); log "docker host $HOSTPOD: $CORES cores, model $SERVED at $MODEL_URL, in flight $INFLIGHT"

# ---- harbor on the host (own venv, python 3.12 via uv)
ssh_h 'export PATH=$HOME/.local/bin:$PATH; command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; export PATH=$HOME/.local/bin:$PATH
  [ -x /root/harborenv/bin/harbor ] || { uv venv /root/harborenv --python 3.12 >/dev/null 2>&1 && VIRTUAL_ENV=/root/harborenv uv pip install -q "harbor==0.21.0" >/dev/null 2>&1; }
  /root/harborenv/bin/harbor --version' | tail -1 | grep -q "0.21" || { log "harbor install failed on $HOSTPOD"; exit 5; }

# ---- the job (same knobs as harbor_cell.py: mini-swe-agent, 4 h / 250 steps, T 0, max_tokens 8192)
JOB=/dev/shm/harbor_jobs; NAME="swe4h-${DIGEST:0:12}"
ssh_h "mkdir -p $JOB"
echo '{"agent": {"step_limit": 250}, "model": {"model_kwargs": {"temperature": 0.0}}}' | ssh_h "cat > $JOB/mswea-$NAME.json"
HCMD="/root/harborenv/bin/harbor run -d swebench-verified@1.0 -a mini-swe-agent -e docker -m openai/$SERVED -n $INFLIGHT -y -q -o $JOB --job-name $NAME --ae OPENAI_API_BASE=$MODEL_URL --ae OPENAI_BASE_URL=$MODEL_URL --ae OPENAI_API_KEY=$KEY --ae MSWEA_API_KEY=$KEY --agent-timeout-multiplier 4.8000 --max-retries 1 --ak max_tokens=8192 --ak config_file=$JOB/mswea-$NAME.json"
echo "$HCMD >> $JOB/$NAME.log 2>&1; echo HARBOR_EXIT=\$? >> $JOB/$NAME.log" | ssh_h "cat > $JOB/run-$NAME.sh"
# job dir on /dev/shm: Lium pods run docker under sysbox and a bind mount from the overlay rootfs fails
# ("error mounting"), tmpfs mounts fine. Images pile up (500 distinct SWE images): prune unused ones every 20 min while the job runs
ssh_h "tmux kill-session -t $NAME 2>/dev/null; tmux new-session -d -s $NAME 'bash $JOB/run-$NAME.sh'; tmux kill-session -t prune-$NAME 2>/dev/null; tmux new-session -d -s prune-$NAME 'while sleep 1200; do docker image prune -af >/dev/null 2>&1; docker network prune -f >/dev/null 2>&1; done'" || { log "could not start harbor"; exit 6; }
log "harbor started in tmux $NAME on $HOSTPOD"
mkdir -p "$CELL"; echo "swe_local.sh: harbor local docker runtime on $HOSTPOD, model $SERVED, started $(date -u +%FT%TZ)" > "$CELL/cmd.txt"

# ---- poll: pull the job dir, summarise, publish the cell as it fills
pull() { rsync -az -e "ssh ${SSHO[*]} -p $H_PORT" --exclude 'agent/*.jsonl' "root@$H_HOST:$JOB/$NAME/" "$CELL/harbor/" 2>/dev/null; }
T0=$(date +%s)
while :; do
  sleep 600; pull
  N=$(ls "$CELL"/harbor/*/result.json 2>/dev/null | wc -l)
  export BENCH_API_KEY="$KEY"
  "$PY" "$HERE/harbor_cell.py" resummarize --env swebench-verified --budget-tag $TAG --model "$SERVED" --model-label king --model-url "$MODEL_URL" --out "$INTO/king" --agent-timeout-s 14400 --step-limit 250 --concurrency "$INFLIGHT" >/dev/null 2>&1 && \
    "$PY" - "$CELL/summary.json" <<'PY'
import json, sys; p = sys.argv[1]; s = json.load(open(p)); s["sandbox"] = "lium-docker"; s["runtime"] = "harbor/docker (Lium box)"; s["harness_note"] = (s.get("harness_note") or "") + "; local docker runtime on a Lium box (Daytona lane full)"; json.dump(s, open(p, "w"), indent=1)
PY
  log "$N/500 trials after $(( ($(date +%s) - T0) / 60 )) min"
  "$PY" "$HERE/publish.py" --run-dir "$INTO" --only-state --partial >/dev/null 2>&1 || true
  if ssh_h "grep -q HARBOR_EXIT $JOB/$NAME.log 2>/dev/null" ; then break; fi
done
pull
"$PY" "$HERE/harbor_cell.py" resummarize --env swebench-verified --budget-tag $TAG --model "$SERVED" --model-label king --model-url "$MODEL_URL" --out "$INTO/king" --agent-timeout-s 14400 --step-limit 250 --concurrency "$INFLIGHT" 2>&1 | tail -1
"$PY" "$HERE/publish.py" --run-dir "$INTO" --only-cells "king/swebench-verified@${TAG}__t0" 2>&1 | tail -1
rm -f "$HERE/state/swe-local-${DIGEST:0:12}.pid"
log "done"
