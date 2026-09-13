#!/bin/bash
# affine-recoverable: the standing teacher-recoverable filter run (pm2 on the
# validator box, cron 13:30 UTC -- before the 14:30 king review and the 16:00
# fold). One run =
#
#   1. candidates.py  -- states for every king_loop_onset / king_pivot turn of
#                        the CURRENT king on a resumable harness that the
#                        side-table affine/state/recoverable/<digest12>.jsonl
#                        does not have yet (fold's own loop labeler + the
#                        king-review pivot table); then the ACP same-task
#                        proxy tasks (Claude Code / pi / Kimi / Hermes king
#                        failures: the teacher replays the whole task); then
#                        RE-RUN states: rows with fewer than
#                        $RECOVERABLE_CONTINUATIONS (3) OK continuations
#                        (`pending_reruns`), unsolved first;
#   2. ship them to the datagen pod(s) ($RECOVERABLE_PODS, default
#                        affine-datagen-2; several = sharded) and run
#                        run_states.py there in a tmux session: <=
#                        $RECOVERABLE_WORKERS containers per pod, 3
#                        continuations per state / task (T = 0.8), no new
#                        continuation after $RECOVERABLE_DEADLINE_HOURS or
#                        $RECOVERABLE_BUDGET_USD (running ones finish, so a
#                        pod is busy <= deadline + 1 h);
#   3. pull the results back and aggregate.py --merge-into the side-table
#                        (existing rows kept, continuations unioned, verdict =
#                        majority of the OK continuations; rewritten atomically).
#
#   ops/recoverable/daily.sh                 # one run (what pm2 executes)
#   RECOVERABLE_DEADLINE_HOURS=0 ops/recoverable/daily.sh   # no cap (backlog)
#   RECOVERABLE_RERUNS_ONLY=1 ops/recoverable/daily.sh      # only top up existing rows
#   RECOVERABLE_NO_RERUNS=1 ops/recoverable/daily.sh        # new states only
#
# Env comes from the same frozen snapshot the other services use
# (~/.affine-validator.env, parsed not sourced; HF_TOKEN for the teacher
# tokenizer the loop labeler bakes with). The pod side reads its own
# /root/recoverable/.env (ENGY_2, the teacher key) through pod_run.sh.
# Everything the job writes lives under affine/state/recoverable/ (gitignored
# state): <digest12>.jsonl side-tables, runs/<stamp>/ (states, results,
# summary), lock (held only while a run merges into the side-table).
set -euo pipefail

REPO="${AFFINE_REPO:-/home/const/subnet120}"
ENV_FILE="${AFFINE_VALIDATOR_ENV:-/home/const/.affine-validator.env}"
PY="$REPO/.venv/bin/python"
OPS="$REPO/ops/recoverable"
STATE="$REPO/affine/state/recoverable"
POD_NAME="${RECOVERABLE_POD:-affine-datagen-2}"
PODS="${RECOVERABLE_PODS:-$POD_NAME}"        # space-separated; sharded
DEADLINE_H="${RECOVERABLE_DEADLINE_HOURS:-6}"
MAX_STATES="${RECOVERABLE_MAX_STATES:-400}"
# 6 containers next to the pod's production batches (24-28 cores, 94 GB):
# 3 continuations per state at ~25 min each is 3x the pre-rule work.
WORKERS="${RECOVERABLE_WORKERS:-6}"
CONTINUATIONS="${RECOVERABLE_CONTINUATIONS:-3}"
# same_task = the ACP same-task proxy (claude_code / pi / kimi_code /
# hermes_agent; states.py PROXY_HARNESSES): 3 fresh teacher rollouts of the
# whole task under the king's harness, admit = solved in a majority, rows
# marked proxy="same_task", capped at RECOVERABLE_ACP_MAX_SHARE of the
# admitted tasks (king-data spec §2.4). Ordered after new resumable states
# and before re-runs, so the deadline bounds it.
KINDS="${RECOVERABLE_KINDS:-textbased,bash,terminus,same_task}"
ACP_MAX_SHARE="${RECOVERABLE_ACP_MAX_SHARE:-0.5}"
# Engy list-price cap per pod run (a same-task ACP rollout is ~10x a
# continuation in tokens); the daily run stops launching past it.
BUDGET_USD="${RECOVERABLE_BUDGET_USD:-40}"
CHUNKS="${RECOVERABLE_CHUNKS:-$REPO/ops/corpus_build/cache/traces/chunks}"
POLL_S="${RECOVERABLE_POLL_S:-300}"
# RECOVERABLE_RETRY_ERRORED=1: errored side-table rows become candidates again
# and errored results in the pod's out dir are re-run (manual use; a 3,600 s
# timeout costs an hour per retry, so the cron run does not do this).
RETRY_FLAG=""
[[ "${RECOVERABLE_RETRY_ERRORED:-0}" == "1" ]] && RETRY_FLAG="--retry-errored"
RERUN_FLAG=""
[[ "${RECOVERABLE_RERUNS_ONLY:-0}" == "1" ]] && RERUN_FLAG="--reruns-only"
[[ "${RECOVERABLE_NO_RERUNS:-0}" == "1" ]] && RERUN_FLAG="--no-reruns"
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
# Runs may overlap (the cron on the current king, a backlog on another pod):
# pods are protected by the per-pod busy check below, the side-table by the
# lock around the merge step (read-modify-write, seconds). Two runs that pick
# the same state land the same continuation twice at worst (merged by trace id).

# RECOVERABLE_KING_DIGEST=<digest12>: work on that king instead of the current
# one (backlog of a previous reign).
DIGEST="${RECOVERABLE_KING_DIGEST:-$("$PY" - "$REPO/affine/state/state.json" <<'PY'
import json, sys
king = json.load(open(sys.argv[1])).get("king") or {}
print(str(king.get("revision") or "")[:12])
PY
)}"
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
  --max-states "$MAX_STATES" --continuations "$CONTINUATIONS" $RETRY_FLAG $RERUN_FLAG \
  2>&1 | tee "$RUN_DIR/candidates.log" | sed "s/^/  /"
N_STATES="$(wc -l < "$RUN_DIR/states/states.jsonl")"
if [[ "$N_STATES" -eq 0 ]]; then
  log "no new states; done"
  exit 0
fi

# 2. pods ---------------------------------------------------------------------
# RECOVERABLE_PODS="affine-datagen-2 affine-datagen-5": the work is split by
# run_states.py --shard i/n over the pods that are free (a pod still running
# a rec-daily session is skipped, not waited for). One pod is the default.
declare -a HOSTS PORTS NAMES
if [[ -n "${RECOVERABLE_POD_SSH:-}" ]]; then
  read -r h p <<<"$RECOVERABLE_POD_SSH" || true
  HOSTS+=("$h"); PORTS+=("$p"); NAMES+=("$POD_NAME")
else
  PODS_TABLE="$("$PY" "$REPO/ops/king-datagen/kingctl.py" pods)"
  for name in $PODS; do
    read -r h p <<<"$(awk -v n="$name" '$1 == n {print $2, $3}' <<<"$PODS_TABLE")" || true
    if [[ -z "${h:-}" || -z "${p:-}" ]]; then
      log "pod $name not found (kingctl.py pods); skipping it"
      continue
    fi
    HOSTS+=("$h"); PORTS+=("$p"); NAMES+=("$name")
  done
fi
if [[ ${#HOSTS[@]} -eq 0 ]]; then
  log "no pod resolved (RECOVERABLE_PODS='$PODS'); set RECOVERABLE_POD_SSH='host port'"
  exit 1
fi
pod_ssh() {  # pod_ssh <index> <remote command...>
  local i="$1"; shift
  ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 \
      -o LogLevel=ERROR -p "${PORTS[$i]}" "root@${HOSTS[$i]}" "$@"
}
N_CONT="$("$PY" -c 'import json,sys
seen=set(); n=0
for l in open(sys.argv[1]):
    s=json.loads(l); k=s.get("proxy_key") or s["state_id"]
    if k in seen: continue
    seen.add(k); n+=int(s.get("continuations_needed") or 0)
print(n)' "$RUN_DIR/states/states.jsonl")"

# Free pods only.
declare -a USE
for i in "${!HOSTS[@]}"; do
  if pod_ssh "$i" "tmux has-session -t rec-daily 2>/dev/null || pgrep -f '[r]un_states.py' >/dev/null"; then
    log "pod ${NAMES[$i]}: a run_states / rec-daily session is still active; skipping it"
  else
    USE+=("$i")
  fi
done
N_PODS=${#USE[@]}
if [[ "$N_PODS" -eq 0 ]]; then
  log "every pod is busy; exiting"
  exit 0
fi
log "$N_STATES state(s) / $N_CONT continuation(s) needed on $N_PODS pod(s), $WORKERS workers each, deadline ${DEADLINE_H} h, budget \$${BUDGET_USD}/pod"

# Code and states to each pod (the pod copy of ops/recoverable follows the box).
shard=0
for i in "${USE[@]}"; do
  log "pod ${NAMES[$i]} = ${HOSTS[$i]}:${PORTS[$i]} shard $shard/$N_PODS"
  tar -C "$REPO/ops" -czf - recoverable | pod_ssh "$i" "mkdir -p $POD_DIR && tar -C $POD_DIR -xzf -"
  tar -C "$RUN_DIR" -czf - states | pod_ssh "$i" "mkdir -p $POD_DIR/daily/$RUN && tar -C $POD_DIR/daily/$RUN -xzf -"
  pod_ssh "$i" "cd $POD_DIR && tmux new-session -d -s rec-daily \
    \"bash ./recoverable/pod_run.sh --states daily/$RUN/states/states.jsonl --out $POD_DIR/daily/out \
       --kinds $KINDS --workers $WORKERS --deadline-hours $DEADLINE_H --continuations $CONTINUATIONS \
       --shard $shard/$N_PODS --budget-usd $BUDGET_USD --untag $RETRY_FLAG \
       > daily/$RUN/run.log 2>&1; touch daily/$RUN/DONE\""
  shard=$((shard + 1))
done

T0=$(date +%s)
if [[ "$DEADLINE_H" == "0" ]]; then HARD_S=0; else
  HARD_S=$(awk -v h="$DEADLINE_H" 'BEGIN {printf "%d", (h + 1.5) * 3600}')
fi
while true; do
  sleep "$POLL_S"
  all_done=1; DONE_N=0
  for i in "${USE[@]}"; do
    pod_ssh "$i" "test -f $POD_DIR/daily/$RUN/DONE" || all_done=0
    n="$(pod_ssh "$i" "grep -c ' -> ' $POD_DIR/daily/$RUN/run.log 2>/dev/null || true")"
    DONE_N=$((DONE_N + ${n:-0}))
  done
  [[ "$all_done" == 1 ]] && break
  ELAPSED=$(( $(date +%s) - T0 ))
  log "running: ${DONE_N}/$N_CONT continuation(s) finished on $N_PODS pod(s), $((ELAPSED / 60)) min"
  if [[ "$HARD_S" -gt 0 && "$ELAPSED" -gt "$HARD_S" ]]; then
    log "hard cap reached; leaving the pod run(s) to finish, merging what is there"
    break
  fi
done

# 3. merge --------------------------------------------------------------------
mkdir -p "$RUN_DIR/out"
for i in "${USE[@]}"; do
  pod_ssh "$i" "cd $POD_DIR/daily/out && tar -czf - --ignore-failed-read results reports" | tar -C "$RUN_DIR/out" -xzf -
  pod_ssh "$i" "cat $POD_DIR/daily/$RUN/run.log" > "$RUN_DIR/run.${NAMES[$i]}.log" || true
done
(
  flock 9   # one merge into the side-table at a time (waits, never skips)
  "$PY" "$OPS/aggregate.py" --states "$RUN_DIR/states/states.jsonl" --out "$RUN_DIR/out" \
    --continuations "$CONTINUATIONS" --acp-max-share "$ACP_MAX_SHARE" --merge-into "$TABLE" \
    | { grep -E "^(merged|states |rule |same-task|side-table)" || true; } | sed "s/^/  /"
) 9>"$STATE/lock"
log "done; summary $RUN_DIR/out/summary.md"
