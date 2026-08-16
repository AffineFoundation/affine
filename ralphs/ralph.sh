#!/usr/bin/env bash
# ralph.sh — hand one GOAL.md to cursor-agent over and over until the goal is met.
#
# Each pass is a FRESH agent with no memory of the last one. The working
# directory is the only thing that carries state between passes, which is what
# makes this work over long horizons: the agent reads the tree, makes one
# increment, and leaves notes for its successor.
#
#   ralph.sh <dir>                 loop on <dir>/GOAL.md, agent runs in <dir>
#   ralph.sh <path/to/GOAL.md>     loop on that file, agent runs in its dir
#
#   -i, --interval S     sleep between passes            (default 10)
#   -t, --timeout S      kill a pass after S seconds     (default 1800)
#   -n, --max-passes N   stop after N passes, 0=forever  (default 0)
#   -m, --model NAME     model, or a comma-separated list to rotate through
#                        when passes stop making progress
#   -b, --background     detach and return immediately
#       --max-stalls N   give up after N unproductive passes in a row
#                        (default 0 = never give up)
#       --stop           stop the background loop for this dir
#       --status         is it running? + recent passes + health
#       --ensure         start it only if it isn't already running
#                        (safe to run from cron every minute)
#
# A pass is judged by whether it CHANGED THE TREE, not by its exit code: a
# model that refuses the prompt, hits a rate limit, or returns an apology all
# exit 0. Unproductive passes are classified (refusal / apierror / stall /
# timeout / fail), logged to .ralph/health.log, and answered with exponential
# backoff plus model rotation. The loop stays alive through them.
#
# Per-run state lives in <dir>/.ralph/:
#   pid  loop.log  status.log  health.log  ATTENTION  passes/NNNN.log  DONE
#
# The loop ends when .ralph/DONE exists (the agent creates it), when
# --max-passes or --max-stalls is reached, or when you stop it.
set -uo pipefail

INTERVAL_S=10
PASS_TIMEOUT=1800
MAX_PASSES=0
MAX_STALLS=0
BACKOFF_MAX=900
MODEL=""
BACKGROUND=0
ACTION=run
TARGET=""

# Only consulted for passes that changed nothing, so loose matching is fine.
REFUSAL_RE="i can[^ ]* (help|assist|follow|do that|comply)|i[^ ]*m (unable|not able|not going)|i won[^ ]*t|cannot comply|against my (system )?(rules|instructions|guidelines)"
APIERR_RE="rate limit|too many requests|429|quota|usage limit|insufficient credit|unauthorized|forbidden|401|403|502|503|econnreset|connection (refused|reset|closed)|network error|not logged in|please (log ?in|authenticate)"

usage() {
  sed -n '2,36p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
  exit "${1:-1}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -i|--interval)   INTERVAL_S="$2"; shift 2 ;;
    -t|--timeout)    PASS_TIMEOUT="$2"; shift 2 ;;
    -n|--max-passes) MAX_PASSES="$2"; shift 2 ;;
    -m|--model)      MODEL="$2"; shift 2 ;;
    --max-stalls)    MAX_STALLS="$2"; shift 2 ;;
    --backoff-max)   BACKOFF_MAX="$2"; shift 2 ;;
    -b|--background) BACKGROUND=1; shift ;;
    --stop)          ACTION=stop; shift ;;
    --status)        ACTION=status; shift ;;
    --ensure)        ACTION=ensure; shift ;;
    --__loop)        ACTION=__loop; shift ;;
    -h|--help)       usage 0 ;;
    -*)              echo "unknown option: $1" >&2; usage ;;
    *)               TARGET="$1"; shift ;;
  esac
done

TARGET="${TARGET:-$PWD}"
if [[ -d "$TARGET" ]]; then
  WORKDIR="$(cd "$TARGET" && pwd)"
  GOAL="$WORKDIR/GOAL.md"
else
  WORKDIR="$(cd "$(dirname "$TARGET")" && pwd)"
  GOAL="$WORKDIR/$(basename "$TARGET")"
fi

STATE="$WORKDIR/.ralph"
PIDFILE="$STATE/pid"
LOGFILE="$STATE/loop.log"
STATUSLOG="$STATE/status.log"
HEALTHLOG="$STATE/health.log"
ATTENTION="$STATE/ATTENTION"
DONEFILE="$STATE/DONE"

now() { date -u +%FT%TZ; }
running() { [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }

case "$ACTION" in
  status)
    running && echo "ON (pid $(cat "$PIDFILE"))" || echo "OFF"
    [[ -f "$DONEFILE" ]] && echo "DONE: $(cat "$DONEFILE")"
    [[ -f "$ATTENTION" ]] && echo "!! ATTENTION: $(cat "$ATTENTION")"
    [[ -f "$STATUSLOG" ]] && { echo "-- recent passes --"; tail -n 8 "$STATUSLOG"; }
    [[ -f "$HEALTHLOG" ]] && { echo "-- health --";        tail -n 8 "$HEALTHLOG"; }
    exit 0 ;;
  stop)
    if running; then
      pid="$(cat "$PIDFILE")"
      # Kill the process group so an in-flight cursor-agent goes down with the loop.
      kill -- "-$pid" 2>/dev/null || kill "$pid"
      rm -f "$PIDFILE"
      echo "stopped (loop pid $pid)"
    else
      rm -f "$PIDFILE"
      echo "not running"
    fi
    # Sweep any agent that escaped the group kill. Matching on the process's cwd
    # is exact — it cannot hit the caller's own shell or an unrelated loop.
    for p in $(pgrep -f "cursor-agent -p --output-format" 2>/dev/null); do
      if [[ "$(readlink -f "/proc/$p/cwd" 2>/dev/null)" == "$WORKDIR" ]]; then
        kill "$p" 2>/dev/null && echo "  also killed stray pass agent pid $p"
      fi
    done
    exit 0 ;;
  ensure)
    if running; then echo "already running (pid $(cat "$PIDFILE"))"; exit 0; fi
    if [[ -f "$DONEFILE" ]]; then echo "not restarting: DONE ($(cat "$DONEFILE"))"; exit 0; fi
    BACKGROUND=1; ACTION=run ;;
esac

[[ -f "$GOAL" ]] || { echo "no goal file: $GOAL" >&2; exit 1; }
mkdir -p "$STATE/passes"

# Model rotation: a single model, or a comma-separated list tried in turn when
# passes stop being productive (one model refusing is not all models refusing).
IFS=',' read -r -a MODELS <<<"${MODEL:-}"
[[ ${#MODELS[@]} -eq 0 ]] && MODELS=("")

agent_ready() {
  command -v cursor-agent >/dev/null || return 2
  cursor-agent status >/dev/null 2>&1 || return 3
  return 0
}

require_agent() {
  agent_ready && return 0
  local rc=$?
  [[ $rc -eq 2 ]] && echo "cursor-agent not on PATH" >&2
  [[ $rc -eq 3 ]] && echo "cursor-agent not logged in (run: cursor-agent login)" >&2
  exit 1
}

# Framing wrapped around the goal text. Keep it short — the goal should dominate.
build_prompt() {
  local n="$1"
  cat <<EOF
You are pass #$n of a Ralph loop: the same goal below is handed to a fresh
agent over and over. You remember nothing from earlier passes. The working
directory is the only state shared between passes, so read it before acting.

Working directory: $WORKDIR

Do one useful increment toward the goal in this pass, then stop. Leave the
tree in a state the next pass can pick up from.

Append exactly one line to .ralph/status.log saying what you did:
  <UTC ISO8601> | pass $n | <what you did>

Create .ralph/DONE (one line saying why) ONLY if the goal is permanently and
completely satisfied. If the goal describes ongoing or repeating work, never
create it — that file stops the loop for good.

--- GOAL ---
$(cat "$GOAL")
EOF
}

status_lines() { [[ -f "$STATUSLOG" ]] && wc -l <"$STATUSLOG" || echo 0; }

# Did this pass touch anything outside the loop's own bookkeeping?
tree_touched() {
  local since="$1"
  [[ -n "$(find "$WORKDIR" -newermt "@$since" \( -type f -o -type d \) \
        -not -path "$STATE" -not -path "$STATE/*" -print -quit 2>/dev/null)" ]]
}

classify() {
  local rc="$1" progressed="$2" log="$3"
  case "$rc" in
    124|137) echo timeout; return ;;
  esac
  [[ "$rc" -ne 0 ]] && { echo "fail-rc$rc"; return; }
  [[ "$progressed" == yes ]] && { echo ok; return; }
  grep -qiE "$REFUSAL_RE" "$log" 2>/dev/null && { echo refusal; return; }
  grep -qiE "$APIERR_RE"  "$log" 2>/dev/null && { echo apierror; return; }
  echo stall
}

run_pass() {
  local n="$1" model="$2" log
  log="$STATE/passes/$(printf '%04d' "$n").log"
  local args=(-p --output-format text --force --trust)
  [[ -n "$model" ]] && args+=(--model "$model")
  # --foreground keeps the agent in the loop's process group. Without it GNU
  # timeout puts it in a new group, and --stop cannot reach it: the pass
  # survives, reparents to init, and races the next pass over the workdir.
  ( cd "$WORKDIR" && timeout --foreground "$PASS_TIMEOUT" cursor-agent "${args[@]}" "$(build_prompt "$n")" ) 2>&1 | tee "$log"
  return "${PIPESTATUS[0]}"
}

loop() {
  local n started stalls=0 rc start_ts progressed outcome model sleep_s lines_before

  # Resume numbering from the pass logs so a restarted loop keeps counting up.
  n="$(find "$STATE/passes" -maxdepth 1 -name '[0-9]*.log' -printf '%f\n' 2>/dev/null \
       | sed 's/\.log$//' | sort -n | tail -1)"
  n="$((10#${n:-0}))"
  started="$n"

  echo "[ralph] start pid=$$ goal=$GOAL interval=${INTERVAL_S}s timeout=${PASS_TIMEOUT}s" \
       "max_passes=$MAX_PASSES max_stalls=$MAX_STALLS models=${MODELS[*]:-default} from_pass=$((n + 1))"

  while true; do
    if [[ -f "$DONEFILE" ]]; then
      echo "[ralph] DONE present, stopping: $(cat "$DONEFILE")"
      break
    fi

    # Auth or PATH can break under a long-running loop; wait it out, don't die.
    if ! agent_ready; then
      echo "[ralph] cursor-agent unavailable (PATH or login) — retrying in ${BACKOFF_MAX}s"
      echo "$(now) | - | agentdown | cursor-agent unavailable" >>"$HEALTHLOG"
      printf 'cursor-agent unavailable since %s — check: cursor-agent status\n' "$(now)" >"$ATTENTION"
      sleep "$BACKOFF_MAX"
      continue
    fi

    n=$((n + 1))
    model="${MODELS[$((stalls % ${#MODELS[@]}))]}"
    start_ts="$(date +%s)"
    lines_before="$(status_lines)"
    echo "[ralph] pass $n start $(now)${model:+ model=$model}"

    run_pass "$n" "$model"
    rc=$?

    progressed=no
    if [[ "$(status_lines)" -gt "$lines_before" ]] || tree_touched "$start_ts"; then
      progressed=yes
    fi
    outcome="$(classify "$rc" "$progressed" "$STATE/passes/$(printf '%04d' "$n").log")"

    echo "$(now) | pass $n | $outcome | rc=$rc dur=$(( $(date +%s) - start_ts ))s progressed=$progressed${model:+ model=$model}" >>"$HEALTHLOG"
    echo "[ralph] pass $n $outcome rc=$rc $(now)"

    if [[ "$outcome" == ok ]]; then
      stalls=0
      rm -f "$ATTENTION"
    else
      stalls=$((stalls + 1))
      echo "$(now) | pass $n | LOOP-$outcome (no progress, streak $stalls)" >>"$STATUSLOG"
      printf '%s: %d unproductive passes in a row (last: %s). See .ralph/health.log\n' \
        "$(now)" "$stalls" "$outcome" >"$ATTENTION"
      if [[ "$MAX_STALLS" -gt 0 && "$stalls" -ge "$MAX_STALLS" ]]; then
        echo "[ralph] $stalls unproductive passes in a row (limit $MAX_STALLS) — stopping"
        break
      fi
    fi

    if [[ "$MAX_PASSES" -gt 0 && "$((n - started))" -ge "$MAX_PASSES" ]]; then
      echo "[ralph] reached max passes ($MAX_PASSES), stopping"
      break
    fi

    # Back off geometrically while unproductive so a rate limit or a refusing
    # model isn't hammered once per interval.
    sleep_s="$INTERVAL_S"
    if [[ "$stalls" -gt 0 ]]; then
      sleep_s=$(( INTERVAL_S * (1 << (stalls > 6 ? 6 : stalls)) ))
      [[ "$sleep_s" -gt "$BACKOFF_MAX" ]] && sleep_s="$BACKOFF_MAX"
      echo "[ralph] backing off ${sleep_s}s (streak $stalls)"
    fi
    sleep "$sleep_s"
  done
  rm -f "$PIDFILE"
}

case "$ACTION" in
  __loop)
    loop ;;
  run)
    running && { echo "already running (pid $(cat "$PIDFILE"))" >&2; exit 1; }
    require_agent
    args=(-i "$INTERVAL_S" -t "$PASS_TIMEOUT" -n "$MAX_PASSES"
          --max-stalls "$MAX_STALLS" --backoff-max "$BACKOFF_MAX")
    [[ -n "$MODEL" ]] && args+=(-m "$MODEL")
    if [[ "$BACKGROUND" == 1 ]]; then
      setsid nohup bash "${BASH_SOURCE[0]}" --__loop "${args[@]}" "$GOAL" >>"$LOGFILE" 2>&1 &
      echo $! >"$PIDFILE"
      echo "ralph ON (pid $(cat "$PIDFILE")) on $GOAL — log: $LOGFILE"
    else
      echo $$ >"$PIDFILE"
      trap 'rm -f "$PIDFILE"' EXIT
      loop
    fi ;;
esac
