#!/bin/bash
# affine-recoverable: the standing teacher-recoverable filter run (pm2 on the
# validator box, cron 13:30 UTC -- before the 14:30 king review and the 16:00
# fold). One run =
#
#   1. candidates.py  -- states for every king_loop_onset / king_pivot turn of
#                        the CURRENT king on a resumable harness that the
#                        side-table affine/state/recoverable/<digest12>.jsonl
#                        does not have yet (fold's own loop labeler + the
#                        king-review pivot table);
#   2. ship them to the datagen pod ($RECOVERABLE_POD, default
#                        affine-datagen-2) and run run_states.py there in a
#                        tmux session: <= $RECOVERABLE_WORKERS containers, no
#                        new state after $RECOVERABLE_DEADLINE_HOURS (running
#                        ones finish, so the pod is busy <= deadline + 1 h);
#   3. pull the results back and aggregate.py --merge-into the side-table
#                        (existing rows kept, rewritten atomically).
#
#   ops/recoverable/daily.sh                 # one run (what pm2 executes)
#   RECOVERABLE_DEADLINE_HOURS=0 ops/recoverable/daily.sh   # no cap (backlog)
#
# Env comes from the same frozen snapshot the other services use
# (~/.affine-validator.env, parsed not sourced; HF_TOKEN for the teacher
# tokenizer the loop labeler bakes with). The pod side reads its own
# /root/recoverable/.env (ENGY_2, the teacher key) through pod_run.sh.
# Everything the job writes lives under affine/state/recoverable/ (gitignored
# state): <digest12>.jsonl side-tables, runs/<stamp>/ (states, results,
# summary), lock. A run that finds another run's lock exits 0 and says so.
set -euo pipefail

REPO="${AFFINE_REPO:-/home/const/subnet120}"
ENV_FILE="${AFFINE_VALIDATOR_ENV:-/home/const/.affine-validator.env}"
PY="$REPO/.venv/bin/python"
OPS="$REPO/ops/recoverable"
STATE="$REPO/affine/state/recoverable"
POD_NAME="${RECOVERABLE_POD:-affine-datagen-2}"
DEADLINE_H="${RECOVERABLE_DEADLINE_HOURS:-6}"
MAX_STATES="${RECOVERABLE_MAX_STATES:-400}"
WORKERS="${RECOVERABLE_WORKERS:-4}"
KINDS="${RECOVERABLE_KINDS:-textbased,bash,terminus}"
CHUNKS="${RECOVERABLE_CHUNKS:-$REPO/ops/corpus_build/cache/traces/chunks}"
POLL_S="${RECOVERABLE_POLL_S:-300}"
POD_DIR=/root/recoverable
RUN="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$STATE/runs/$RUN"

log() { echo "$(date -u +%FT%TZ) recoverable[$RUN] $*"; }

if [[ -r "$ENV_FILE" ]]; then
  # KEY=VALUE lines of the pm2 env snapshot; pm2/shell noise skipped.
  eval "$("$PY" - "$ENV_FILE" <<'PY'
import re, shlex, sys
skip_prefix = ("PM2_", "NODE_", "PS1", "OLDPWD", "SHLVL", "_")
for line in open(sys.argv[1]):
    line = line.rstrip("\n")
    m = re.match(r"^([A-Z][A-Z0-9_]*)=(.*)$", line)
    if not m or m.group(1).startswith(skip_prefix):
        continue
    print(f"export {m.group(1)}={shlex.quote(m.group(2))}")
PY
)"
else
  log "no env snapshot at $ENV_FILE (continuing; tokenizer must be cached)"
fi

mkdir -p "$STATE/runs"
exec 9>"$STATE/lock"
if ! flock -n 9; then
  log "another run holds $STATE/lock; exiting"
  exit 0
fi

DIGEST="$("$PY" - "$REPO/affine/state/state.json" <<'PY'
import json, sys
king = json.load(open(sys.argv[1])).get("king") or {}
print(str(king.get("revision") or "")[:12])
PY
)"
if [[ -z "$DIGEST" ]]; then
  log "no king in affine/state/state.json; nothing to do"
  exit 0
fi
PIVOTS="$REPO/affine/state/king_pivots/$DIGEST.jsonl"
TABLE="$STATE/$DIGEST.jsonl"
mkdir -p "$RUN_DIR"
ln -sfn "$RUN_DIR" "$STATE/runs/latest"
log "king $DIGEST; pivots $PIVOTS; side-table $TABLE ($( [[ -f "$TABLE" ]] && wc -l < "$TABLE" || echo 0 ) rows)"

# 1. candidates ---------------------------------------------------------------
"$PY" "$OPS/candidates.py" --chunks "$CHUNKS" --pivots "$PIVOTS" --side-table "$TABLE" \
  --king-digest "$DIGEST" --out "$RUN_DIR/states" --kinds "$KINDS" \
  --max-states "$MAX_STATES" 2>&1 | tee "$RUN_DIR/candidates.log" | sed "s/^/  /"
N_STATES="$(wc -l < "$RUN_DIR/states/states.jsonl")"
if [[ "$N_STATES" -eq 0 ]]; then
  log "no new states; done"
  exit 0
fi

# 2. pod ----------------------------------------------------------------------
if [[ -n "${RECOVERABLE_POD_SSH:-}" ]]; then
  read -r POD_HOST POD_PORT <<<"$RECOVERABLE_POD_SSH" || true
else
  read -r POD_HOST POD_PORT <<<"$("$PY" "$REPO/ops/king-datagen/kingctl.py" pods \
    | awk -v n="$POD_NAME" '$1 == n {print $2, $3}')" || true
fi
if [[ -z "${POD_HOST:-}" || -z "${POD_PORT:-}" ]]; then
  log "pod $POD_NAME not found (kingctl.py pods); set RECOVERABLE_POD_SSH='host port'"
  exit 1
fi
SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20
     -o LogLevel=ERROR -p "$POD_PORT" "root@$POD_HOST")
log "pod $POD_NAME = $POD_HOST:$POD_PORT; $N_STATES state(s), $WORKERS workers, deadline ${DEADLINE_H} h"

if "${SSH[@]}" "tmux has-session -t rec-daily 2>/dev/null || pgrep -f '[r]un_states.py' >/dev/null"; then
  log "a run_states / rec-daily session is still active on the pod; exiting"
  exit 0
fi
# Code and states to the pod (the pod copy of ops/recoverable follows the box).
tar -C "$REPO/ops" -czf - recoverable | "${SSH[@]}" "mkdir -p $POD_DIR && tar -C $POD_DIR -xzf -"
tar -C "$RUN_DIR" -czf - states | "${SSH[@]}" "mkdir -p $POD_DIR/daily/$RUN && tar -C $POD_DIR/daily/$RUN -xzf -"
"${SSH[@]}" "cd $POD_DIR && tmux new-session -d -s rec-daily \
  \"bash ./recoverable/pod_run.sh --states daily/$RUN/states/states.jsonl --out $POD_DIR/daily/out \
     --kinds $KINDS --workers $WORKERS --deadline-hours $DEADLINE_H --untag \
     > daily/$RUN/run.log 2>&1; touch daily/$RUN/DONE\""

T0=$(date +%s)
if [[ "$DEADLINE_H" == "0" ]]; then HARD_S=0; else
  HARD_S=$(awk -v h="$DEADLINE_H" 'BEGIN {printf "%d", (h + 1.5) * 3600}')
fi
while true; do
  sleep "$POLL_S"
  if "${SSH[@]}" "test -f $POD_DIR/daily/$RUN/DONE"; then break; fi
  ELAPSED=$(( $(date +%s) - T0 ))
  DONE_N="$("${SSH[@]}" "grep -c ' -> ' $POD_DIR/daily/$RUN/run.log 2>/dev/null || true")"
  log "running: ${DONE_N:-0}/$N_STATES finished, $((ELAPSED / 60)) min"
  if [[ "$HARD_S" -gt 0 && "$ELAPSED" -gt "$HARD_S" ]]; then
    log "hard cap reached; leaving the pod run to finish, merging what is there"
    break
  fi
done

# 3. merge --------------------------------------------------------------------
mkdir -p "$RUN_DIR/out"
"${SSH[@]}" "cd $POD_DIR/daily/out && tar -czf - --ignore-failed-read results reports" | tar -C "$RUN_DIR/out" -xzf -
"${SSH[@]}" "cat $POD_DIR/daily/$RUN/run.log" > "$RUN_DIR/run.log" || true
"$PY" "$OPS/aggregate.py" --states "$RUN_DIR/states/states.jsonl" --out "$RUN_DIR/out" \
  --merge-into "$TABLE" | { grep -E "^(merged|states |side-table)" || true; } | sed "s/^/  /"
log "done; summary $RUN_DIR/out/summary.md"
