#!/bin/bash
# For one model: wait for its serving box, write a per-model king env on the
# backfill pod, start the coverage driver there in its own tmux session and
# data dir, then (in the background) release the box when the driver is done.
#   start_env_backfill.sh <pod> <digest12> <label> "<sources|all>" <max_containers>
set -uo pipefail
cd ~/subnet120/ops/benchsuite
HERE=$PWD; REPO=~/subnet120; PY=$REPO/.venv/bin/python
source ./env.sh
export LIUM_API_KEY="${LIUM_API_KEY:-${LIUM:-}}"
POD="$1"; D12="$2"; LABEL="$3"; SOURCES="$4"; MAXC="${5:-12}"
BF_HOST=73.139.34.205; BF_PORT=20009
BF=(ssh -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -p $BF_PORT root@$BF_HOST)
LOG=~/subnet120/ops/coverage/state/env_backfill_$D12.log
log() { echo "[env_backfill $LABEL] $(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
$PY kingpod.py wait "$POD" > /dev/null 2>&1 || { log "box $POD never became ready; releasing"; $PY kingpod.py release "$POD"; exit 3; }
read -r URL KEY SERVED DIGEST < <($PY -c 'import json; m=json.load(open("state/pods.json"))["'"$POD"'"]; print(m["base_url"], m["key"], m["served"], m.get("digest",""))')
log "box $POD ready: $URL serving $SERVED"
"${BF[@]}" "cat > /root/rollouts/.king_env_$D12 <<EOF
# written by the coverage worker for the env backfill ($LABEL, box $POD)
KING_BASE_URL=$URL
KING_MODEL=$SERVED
KING_KEY=$KEY
KING_DIGEST=$DIGEST
KING_REIGN=$LABEL
EOF
cat > /root/rollouts/run_backfill_$D12.sh <<'EOF2'
#!/bin/bash
set -uo pipefail
cd /root/rollouts
source /root/affine/.datagen_env; source /root/rollouts/.rollouts_env
export PATH=/root/.local/bin:\$PATH PYTHONPATH=/root/affine:/root/rollouts
export ROLLOUTS_DATA_DIR=/root/rollouts-data-$D12 ROLLOUTS_KING_ENV=/root/rollouts/.king_env_$D12 ROLLOUTS_MAX_CONTAINERS=$MAXC ROLLOUTS_BATCH_SIZE=$MAXC
[ \"\${ROLLOUTS_R2_PREFIX:-}\" = traces-backfill/ ] || { echo \"refusing: ROLLOUTS_R2_PREFIX=\${ROLLOUTS_R2_PREFIX:-}\"; exit 2; }
exec /root/venv/bin/python -m rollouts.backfill \"\$@\"
EOF2
chmod +x /root/rollouts/run_backfill_$D12.sh
mkdir -p /root/rollouts-data-$D12
tmux -f /dev/null new-session -d -s backfill-$D12 \"/root/rollouts/run_backfill_$D12.sh --digest12 $D12 --n 50 --sources '$SOURCES' >> /root/logs/backfill_$D12.log 2>&1\"
sleep 3; tmux ls | grep backfill-$D12; tail -2 /root/logs/backfill_$D12.log"
log "driver started on affine-backfill-2 (tmux backfill-$D12, containers $MAXC, sources: $SOURCES)"
# release the box once the driver finishes (poll every 10 min; 15-min rule)
(
  while :; do
    sleep 600
    if "${BF[@]}" "grep -q 'backfill $D12 complete' /root/logs/backfill_$D12.log 2>/dev/null || ! tmux -f /dev/null has-session -t backfill-$D12 2>/dev/null"; then
      log "driver finished; releasing $POD"
      $PY kingpod.py release "$POD" >> "$LOG" 2>&1
      $PY - ~/subnet120/ops/coverage/state/backfill_pods.json "$POD" <<'PY'
import json, sys, time
p, pod = sys.argv[1:]
rows = json.load(open(p))
for r in rows:
    if r["pod"] == pod and not r.get("released_at"):
        r["released_at"] = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
json.dump(rows, open(p, "w"), indent=1)
PY
      break
    fi
  done
) > /dev/null 2>&1 &
disown
log "release watcher pid $!"
