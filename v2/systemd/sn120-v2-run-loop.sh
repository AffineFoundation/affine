#!/usr/bin/env bash
# SN120 v2: the restart loop of the round runner on the GPU box.
#
#   sn120-v2-run-loop.sh PYTHON --state-dir DIR --stack MODULE:FUNCTION [more arguments of "run"]
#
# PYTHON is the python of the virtual environment where sn120_v2 is installed. Every argument after it goes to
# "PYTHON -m sn120_v2 run" as it is. The variables that "run" needs (SN120_R2_*, SN120_WORK_DIR, ...) come from
# the environment. sn120-v2-run.service starts this same script, so a box with systemd and a box without it do
# the same.
#
# Before each start of "run", in this order:
#   1. Stop every process that the last "run" left (model servers, training jobs), and wait until no GPU holds
#      memory. A GPU that stays in use stops the loop: a start on a used GPU fails the round's training.
#   2. End an open, undecided round with "end-round": no decision, nobody is charged, no failure is counted.
#      Without this step, a start inside the training time of a round can count a failure for a seat. If a box
#      result of that round is already in the store, the round is not ended: "run" judges that result.
#   3. Start "run".
#
# After "run" ends, by its exit code:
#   0                a finished run ("--rounds", "--max-idle"): the loop stops with 0
#   2                "run" refused to start (parameters, state, a journal that another controller took): the loop
#                    stops with 2. One exception: "refused: InfraError" (the store or the chain did not answer).
#                    The loop waits and tries again, without end: see below.
#   130, 143         "run" was stopped with SIGINT or SIGTERM: an operator's stop, the loop stops with 0
#   every other code a crash (1 is a Python error, 137 is SIGKILL or out of memory): the loop starts "run" again
# A signal to the loop itself (SIGTERM, SIGINT, SIGHUP) is an operator's stop: the loop sends SIGTERM to "run",
# stops the servers and jobs that it left, and stops with 0. "run" is safe to stop at any point: the journal is
# written to the store first, and the next start continues from it.
#
# The loop counts the starts with no finished round between them. A round that "run" closed sets the count back
# to zero. At SN120_LOOP_MAX_STARTS the loop stops with 3; the last lines of the log say why and what to run.
#
# A try that ends with "refused: InfraError", from "end-round" or from "run", is not in that count, and it does
# not set the count back to zero. A network fault of the box at night must not stop the loop: it tries again
# until the store and the chain answer. The pause doubles with each such try, from SN120_LOOP_PAUSE_S up to
# SN120_LOOP_MAX_PAUSE_S. The log and run-loop.status say what the loop waits for and since when.
# Exit codes of the loop: 0 stopped or finished, 2 refused, 3 gave up, 4 a GPU stays in use, 64 wrong call.
#
# Files, all in $SN120_WORK_DIR/logs:
#   run.log           the output of "run" and the lines of this loop ("... run-loop: ...")
#   run-loop.status   one line: what the loop does now, or why it stopped
#   run-loop.pid      the process id of the loop while it runs
#   run-loop.lock     one loop for each work directory
#
# Settings (environment, all optional):
#   SN120_LOOP_PAUSE_S        seconds between the end of a "run" and the next start (30)
#   SN120_LOOP_MAX_STARTS     starts in a row with no finished round, then the loop stops (5)
#   SN120_LOOP_MAX_PAUSE_S    the longest pause between two tries while the store or the chain does not answer (300)
#   SN120_LOOP_STOP_GRACE_S   seconds that "run" gets after SIGTERM at an operator's stop, then SIGKILL (30)
#   SN120_LOOP_KILL_GRACE_S   seconds that a left process gets after SIGTERM, then SIGKILL (30)
#   SN120_LOOP_GPU_FREE_MIB   a GPU with more used memory than this is in use (1024, the limit of the evaluator)
#   SN120_LOOP_GPU_WAIT_S     seconds to wait for free GPUs before the loop stops (120)

set -u

usage() {
    echo "sn120-v2-run-loop: $1" >&2
    echo "usage: sn120-v2-run-loop.sh PYTHON --state-dir DIR --stack MODULE:FUNCTION [more arguments of run]" >&2
    exit 64
}

[ $# -ge 1 ] || usage "no arguments"
PY=$1
shift
RUN_ARGS=("$@")
STATE_DIR=""
STACK=""
for ((i = 0; i < ${#RUN_ARGS[@]}; i++)); do
    case ${RUN_ARGS[i]} in
        --state-dir) STATE_DIR=${RUN_ARGS[i + 1]:-} ;;
        --state-dir=*) STATE_DIR=${RUN_ARGS[i]#*=} ;;
        --stack) STACK=${RUN_ARGS[i + 1]:-} ;;
        --stack=*) STACK=${RUN_ARGS[i]#*=} ;;
    esac
done
[ -n "$STATE_DIR" ] && [ -n "$STACK" ] || usage "--state-dir and --stack are required"
[ -d "$STATE_DIR" ] || usage "the state directory $STATE_DIR does not exist. On a new box make it empty (mkdir -p): run pulls the journal copy from the store."
[ -n "${SN120_WORK_DIR:-}" ] || usage "SN120_WORK_DIR is not set"
case $PY in
    */*) PATH="$(dirname "$PY"):$PATH" && export PATH ;;  # as an active virtual environment does
esac
command -v "$PY" > /dev/null || usage "PYTHON is not a program: $PY"

PAUSE_S=${SN120_LOOP_PAUSE_S:-30}
MAX_STARTS=${SN120_LOOP_MAX_STARTS:-5}
MAX_PAUSE_S=${SN120_LOOP_MAX_PAUSE_S:-300}
STOP_GRACE_S=${SN120_LOOP_STOP_GRACE_S:-30}
KILL_GRACE_S=${SN120_LOOP_KILL_GRACE_S:-30}
GPU_FREE_MIB=${SN120_LOOP_GPU_FREE_MIB:-1024}
GPU_WAIT_S=${SN120_LOOP_GPU_WAIT_S:-120}

LOGS=$SN120_WORK_DIR/logs
LOG=$LOGS/run.log
STATUS_FILE=$LOGS/run-loop.status
PID_FILE=$LOGS/run-loop.pid
mkdir -p "$LOGS" || usage "cannot make $LOGS"

# "run" gets this variable, and every process that it starts inherits it. That is how the loop finds what a dead
# "run" left: those processes run in their own sessions and outlive it.
MARK="SN120_LOOP_MARK=$(cd "$STATE_DIR" && pwd -P)"
SHOW_STATUS="$PY -m sn120_v2 status --state-dir $STATE_DIR"
END_ROUND="$PY -m sn120_v2 end-round --state-dir $STATE_DIR --stack $STACK"

exec 9> "$LOGS/run-loop.lock"
flock -n 9 || usage "another loop runs with this work directory (see $PID_FILE)"

RUN_PID=""
WATCHDOG=""
STOPPING=""
NO_SMI_SAID=""

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }

say() {
    local line
    line="$(stamp) run-loop: $*"
    echo "$line" >> "$LOG"
    echo "$line" >&2
}

status() { echo "$(stamp) $*" > "$STATUS_FILE"; }

nap() {  # a sleep that a signal to the loop ends at once
    sleep "$1" 9>&- &
    wait $! 2> /dev/null
}

on_signal() {
    [ -z "$STOPPING" ] || return 0
    STOPPING=$1
    if [ -n "$RUN_PID" ]; then
        say "SIG$1: an operator's stop. Sending SIGTERM to run (pid $RUN_PID)."
        kill -TERM "$RUN_PID" 2> /dev/null
        (sleep "$STOP_GRACE_S" && kill -KILL "$RUN_PID" 2> /dev/null) 9>&- > /dev/null 2>&1 &
        WATCHDOG=$!
    else
        say "SIG$1: an operator's stop."
    fi
}
trap 'on_signal INT' INT
trap 'on_signal TERM' TERM
trap 'on_signal HUP' HUP

marked() {  # the process ids of everything that a run of this loop started
    local dir pid
    for dir in /proc/[0-9]*; do
        pid=${dir#/proc/}
        [ "$pid" = "$$" ] && continue
        grep -qzxF -- "$MARK" "$dir/environ" 2> /dev/null && echo "$pid"
    done
}

stop_leftovers() {  # 0 when no process of an earlier run is left
    local pids waited=0
    pids=$(marked)
    [ -n "$pids" ] || return 0
    say "stopping what the last run left: pids" $pids
    kill -TERM $pids 2> /dev/null
    while [ -n "$(marked)" ] && [ "$waited" -lt "$KILL_GRACE_S" ]; do
        sleep 1
        waited=$((waited + 1))
    done
    pids=$(marked)
    if [ -n "$pids" ]; then
        say "still there after $KILL_GRACE_S s, sending SIGKILL: pids" $pids
        kill -KILL $pids 2> /dev/null
        sleep 2
    fi
    LEFT=$(marked)
    [ -z "$LEFT" ]
}

busy_gpus() {  # prints the GPUs in use; fails when the check itself fails
    local out
    out=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>&1) || {
        echo "nvidia-smi failed: ${out##*$'\n'}"
        return 1
    }
    awk -F', *' -v limit="$GPU_FREE_MIB" '$2 + 0 > limit { printf "%s ", $1 }' <<< "$out"
}

wait_gpus_free() {  # 0 when no GPU is in use; GPU_PROBLEM says what is wrong otherwise
    local waited=0 busy
    if ! command -v nvidia-smi > /dev/null; then
        [ -n "$NO_SMI_SAID" ] || say "no nvidia-smi on this machine: the GPU check is skipped"
        NO_SMI_SAID=1
        return 0
    fi
    while :; do
        if busy=$(busy_gpus); then
            [ -z "$busy" ] && return 0
            GPU_PROBLEM="GPUs ${busy}hold more than $GPU_FREE_MIB MiB"
        else
            GPU_PROBLEM=$busy
        fi
        [ "$waited" -lt "$GPU_WAIT_S" ] && [ -z "$STOPPING" ] || return 1
        nap 5
        waited=$((waited + 5))
    done
}

rounds_closed() {  # reads the local journal only; takes nothing over
    "$PY" -m sn120_v2 status --state-dir "$STATE_DIR" 2> /dev/null | sed -n 's/^ *"rounds_closed": *\([0-9][0-9]*\).*/\1/p' | head -n 1
}

finish() {  # finish CODE STATE LINE...: the last lines of the log say why the loop stops and what to run next
    local code=$1 state=$2 line
    shift 2
    stop_leftovers || say "could not stop these processes of the last run: pids" $LEFT
    for line in "$@"; do
        say "$line"
    done
    status "$state (exit code $code of the loop)"
    rm -f "$PID_FILE"
    exit "$code"
}

stopped_by_operator() {
    finish 0 "stopped by the operator" \
        "STOPPED by the operator${1:+ ($1)}. Nothing starts run again." \
        "To go on, start the loop again. It frees the GPUs, ends an open round with no decision (nobody is charged, no failure is counted; a stored box result is kept and judged) and starts run."
}

refused() {  # refused WHAT LINE
    finish 2 "stopped: $1 refused" \
        "STOPPED: $1 refused, and a new start cannot change that: ${2:-no refused line in the log}" \
        "Fix what that line names, then start the loop again. FencedError means that another process took the journal: only one box, and one loop on it, may run the rounds. See the state: $SHOW_STATUS"
}

gave_up() {
    finish 3 "stopped: gave up after $MAX_STARTS starts with no finished round" \
        "STOPPED: $MAX_STARTS starts in a row and no round finished. The last one ended with: $1" \
        "The fault is in the lines above, in $LOG. See the state: $SHOW_STATUS" \
        "After the fix, start the loop again. It ends an open round with no decision (nobody is charged, no failure is counted). A stored box result is kept: run judges it, with no new round." \
        "Only if that stored result itself is bad: $END_ROUND --discard-result   (ends the round with no decision and THROWS THE RESULT AWAY; nobody is charged)"
}

end_open_round() {  # 0: go on. 1: the command failed, FAILURE says how (INFRA: with "refused: InfraError"). Stops the loop when the command is refused.
    local out rc
    out=$("$PY" -m sn120_v2 end-round --state-dir "$STATE_DIR" --stack "$STACK" 2>&1 9>&-)
    rc=$?
    case $rc in
        0) say "end-round: the open round is ended with no decision. Nobody is charged and no failure is counted." ;;
        3) say "end-round: a box result of the open round is in the store. The round stays open, and run judges that result." ;;
        2)
            case $out in
                *"refused: V2Error: no round is open"*) ;;
                *"refused: InfraError:"*)
                    echo "$out" >> "$LOG"
                    FAILURE="end-round: $(grep -m 1 '^refused: ' <<< "$out")"
                    INFRA=1
                    return 1
                    ;;
                *)
                    echo "$out" >> "$LOG"
                    refused "end-round" "$(grep -m 1 '^refused: ' <<< "$out")"
                    ;;
            esac
            ;;
        *)
            echo "$out" >> "$LOG"
            FAILURE="end-round ended with exit code $rc"
            return 1
            ;;
    esac
    return 0
}

run_once() {  # 0: start again. Stops the loop for every exit code that means "do not start again".
    local offset rc again line
    offset=$(stat -c %s "$LOG" 2> /dev/null || echo 0)
    (
        trap - INT TERM HUP
        exec 9>&-
        exec setsid env "$MARK" "$PY" -m sn120_v2 run "${RUN_ARGS[@]}"
    ) >> "$LOG" 2>&1 < /dev/null &
    RUN_PID=$!
    status "running: run has pid $RUN_PID, start $((STARTS + 1)) of at most $MAX_STARTS with no finished round"
    while :; do
        wait "$RUN_PID" 2> /dev/null
        rc=$?
        kill -0 "$RUN_PID" 2> /dev/null && continue  # a signal to the loop ended the wait, not the end of run
        wait "$RUN_PID" 2> /dev/null
        again=$?
        [ "$again" = 127 ] || rc=$again
        break
    done
    RUN_PID=""
    [ -z "$WATCHDOG" ] || kill "$WATCHDOG" 2> /dev/null
    [ -z "$STOPPING" ] || stopped_by_operator "SIG$STOPPING to the loop; run ended with exit code $rc"
    line=$(tail -c "+$((offset + 1))" "$LOG" | grep -m 1 '^refused: ')
    case $rc in
        0)
            finish 0 "stopped: run finished" \
                "STOPPED: run finished with exit code 0 (--rounds or --max-idle). Nothing starts it again." \
                "For more rounds, start the loop again."
            ;;
        130 | 143) stopped_by_operator "run ended with exit code $rc: SIGINT or SIGTERM" ;;
        2)
            case $line in
                "refused: InfraError:"*) FAILURE="exit code 2, $line" && INFRA=1 ;;
                *) refused "run" "$line" ;;
            esac
            ;;
        1) FAILURE="exit code 1 (a Python error: the traceback is above)" ;;
        137) FAILURE="exit code 137 (SIGKILL: out of memory, or a kill by hand)" ;;
        *) FAILURE="exit code $rc" ;;
    esac
    return 0
}

echo "$$" > "$PID_FILE"
say "loop started (pid $$): $PY -m sn120_v2 run ${RUN_ARGS[*]}"
STARTS=0
INFRA_SINCE=""
INFRA_TRIES=0
INFRA_PAUSE=$PAUSE_S
while :; do
    status "before start: freeing the GPUs"
    if ! stop_leftovers; then
        finish 4 "stopped: processes of the last run cannot be stopped" \
            "STOPPED: processes of the last run do not end, also not after SIGKILL: pids $LEFT" \
            "Stop them by hand (kill -9 as root, or restart the box), then start the loop again."
    fi
    if ! wait_gpus_free; then
        [ -z "$STOPPING" ] || stopped_by_operator "SIG$STOPPING to the loop"
        finish 4 "stopped: a GPU stays in use" \
            "STOPPED: run was not started, because after $GPU_WAIT_S s: $GPU_PROBLEM. No process of this loop is left, so another process uses the GPUs." \
            "Find it: nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv   Stop it (a run that was started by hand, its model servers and training jobs), then start the loop again."
    fi
    [ -z "$STOPPING" ] || stopped_by_operator "SIG$STOPPING to the loop"
    FAILURE=""
    INFRA=""
    CLOSED=0
    if end_open_round; then
        [ -z "$STOPPING" ] || stopped_by_operator "SIG$STOPPING to the loop"
        BEFORE=$(rounds_closed)
        say "start $((STARTS + 1)) of at most $MAX_STARTS with no finished round between them (rounds closed so far: ${BEFORE:-unknown})"
        run_once
        AFTER=$(rounds_closed)
        [ -z "$BEFORE" ] || [ -z "$AFTER" ] || [ "$AFTER" -le "$BEFORE" ] || CLOSED=$((AFTER - BEFORE))
        ENDED="run ended with $FAILURE"
    else
        ENDED="$FAILURE. run was not started"
    fi
    if [ "$CLOSED" -gt 0 ]; then
        say "$ENDED. It closed $CLOSED round(s) in this start: the count of starts goes back to zero."
        STARTS=0
    fi
    if [ -n "$INFRA" ]; then
        # The store or the chain did not answer. That is not a fault of run: the try is not counted, and the loop never gives up on it.
        [ -n "$INFRA_SINCE" ] || INFRA_SINCE=$(stamp)
        INFRA_TRIES=$((INFRA_TRIES + 1))
        [ "$CLOSED" -gt 0 ] || say "$ENDED."
        say "The store or the chain did not answer: try $INFRA_TRIES, no answer since $INFRA_SINCE. This try is not counted (the count of starts stays at $STARTS of at most $MAX_STARTS). The loop tries again until they answer. Next try in $INFRA_PAUSE s."
        say "A cause on this box (a wrong store token, a wrong package version) does not go away by itself: then stop the loop and fix what the refused line names."
        status "waiting for the store or the chain since $INFRA_SINCE ($INFRA_TRIES tries with no answer); next try in $INFRA_PAUSE s; the last one ended with: $FAILURE"
        nap "$INFRA_PAUSE"
        [ -z "$STOPPING" ] || stopped_by_operator "SIG$STOPPING to the loop"
        INFRA_PAUSE=$((INFRA_PAUSE * 2))
        [ "$INFRA_PAUSE" -le "$MAX_PAUSE_S" ] || INFRA_PAUSE=$((MAX_PAUSE_S > PAUSE_S ? MAX_PAUSE_S : PAUSE_S))
        continue
    fi
    INFRA_SINCE=""
    INFRA_TRIES=0
    INFRA_PAUSE=$PAUSE_S
    if [ "$CLOSED" -eq 0 ]; then
        STARTS=$((STARTS + 1))
        say "$ENDED. No round finished: this counts as start $STARTS of at most $MAX_STARTS."
    fi
    [ "$STARTS" -lt "$MAX_STARTS" ] || gave_up "$FAILURE"
    status "waiting $PAUSE_S s before the next start; the last one ended with: $FAILURE"
    say "next start in $PAUSE_S s"
    nap "$PAUSE_S"
    [ -z "$STOPPING" ] || stopped_by_operator "SIG$STOPPING to the loop"
done
