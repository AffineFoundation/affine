#!/usr/bin/env bash
# Generic cursor-agent loop runner ("ralphs"). Each loop lives in ralphs/<name>/:
#
#   prompt.md    the prompt run by cursor-agent each pass   (required)
#   loop.conf    optional bash-sourced tuning:
#                  INTERVAL_S=900      sleep between passes
#                  PASS_TIMEOUT=2700   timeout per pass, seconds
#                  MODEL=""            cursor-agent --model override
#   loop.log     loop + full agent output       (written by this script)
#   pid          loop pidfile                   (written by this script)
#   status.log   one line per pass              (written by the agent, by
#                convention — tell it to in the prompt)
#
#   ./ralphs/ralphctl.sh <name> on        start the loop (background)
#   ./ralphs/ralphctl.sh <name> off       stop the loop
#   ./ralphs/ralphctl.sh <name> once      single pass in the foreground
#   ./ralphs/ralphctl.sh <name> status    running? + last status lines
#   ./ralphs/ralphctl.sh list             all loops and their state
#
# Env vars KEEPALIVE-style overrides also work at 'on'/'once':
#   LOOP_INTERVAL_S / LOOP_PASS_TIMEOUT / LOOP_MODEL beat loop.conf.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOOPS_DIR="$ROOT/ralphs"

usage() { echo "usage: $0 <name> on|off|once|status  |  $0 list" >&2; exit 1; }

if [[ "${1:-}" == "list" ]]; then
  [[ -d "$LOOPS_DIR" ]] || { echo "no loops defined ($LOOPS_DIR missing)"; exit 0; }
  for d in "$LOOPS_DIR"/*/; do
    [[ -f "$d/prompt.md" ]] || continue
    name="$(basename "$d")"
    if [[ -f "$d/pid" ]] && kill -0 "$(cat "$d/pid")" 2>/dev/null; then
      echo "$name: ON (pid $(cat "$d/pid"))"
    else
      echo "$name: OFF"
    fi
  done
  exit 0
fi

NAME="${1:-}"; CMD="${2:-status}"
[[ -n "$NAME" ]] || usage
DIR="$LOOPS_DIR/$NAME"
PROMPT="$DIR/prompt.md"
CONF="$DIR/loop.conf"
PIDFILE="$DIR/pid"
LOGFILE="$DIR/loop.log"
STATUSLOG="$DIR/status.log"

[[ -f "$PROMPT" ]] || { echo "no such loop: $NAME ($PROMPT missing)" >&2; exit 1; }

# Defaults < loop.conf < LOOP_* env overrides.
INTERVAL_S=900
PASS_TIMEOUT=2700
MODEL=""
# shellcheck source=/dev/null
[[ -f "$CONF" ]] && source "$CONF"
INTERVAL_S="${LOOP_INTERVAL_S:-$INTERVAL_S}"
PASS_TIMEOUT="${LOOP_PASS_TIMEOUT:-$PASS_TIMEOUT}"
MODEL="${LOOP_MODEL:-$MODEL}"

running() { [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }

check_agent() {
  command -v cursor-agent >/dev/null || { echo "cursor-agent not on PATH" >&2; exit 1; }
  cursor-agent status >/dev/null 2>&1 || { echo "cursor-agent not logged in (run: cursor-agent login)" >&2; exit 1; }
}

agent_pass() {
  local model_args=()
  [[ -n "$MODEL" ]] && model_args=(--model "$MODEL")
  # --trust: non-interactive runs require workspace trust.
  # --force: headless passes must be able to run shell commands (curl, doppler);
  #          safety comes from each loop's prompt.md hard rules, not approvals.
  # timeout guards against a hung pass blocking the loop forever.
  timeout "$PASS_TIMEOUT" cursor-agent -p "$(cat "$PROMPT")" \
    --output-format text --trust --force "${model_args[@]}"
}

loop() {
  echo "[$NAME] loop start pid=$$ interval=${INTERVAL_S}s timeout=${PASS_TIMEOUT}s"
  while true; do
    echo "[$NAME] pass start $(date -u +%FT%TZ)"
    if agent_pass; then
      echo "[$NAME] pass ok $(date -u +%FT%TZ)"
    else
      rc=$?
      echo "[$NAME] pass FAILED rc=$rc $(date -u +%FT%TZ)"
      echo "$(date -u +%FT%TZ) | pass-error rc=$rc | agent run failed or timed out" >>"$STATUSLOG"
    fi
    sleep "$INTERVAL_S"
  done
}

case "$CMD" in
  on)
    if running; then
      echo "already ON (pid $(cat "$PIDFILE"))"
      exit 0
    fi
    check_agent
    cd "$ROOT"
    nohup bash "$ROOT/ralphs/ralphctl.sh" "$NAME" __loop >>"$LOGFILE" 2>&1 &
    echo $! >"$PIDFILE"
    echo "$NAME ON (pid $(cat "$PIDFILE")) — every ${INTERVAL_S}s; log: $LOGFILE"
    ;;
  __loop)
    loop
    ;;
  off)
    if running; then
      pid="$(cat "$PIDFILE")"
      kill "$pid"
      rm -f "$PIDFILE"
      echo "$NAME OFF (killed loop pid $pid; an in-flight agent pass may finish)"
    else
      rm -f "$PIDFILE"
      echo "not running"
    fi
    ;;
  once)
    check_agent
    cd "$ROOT"
    agent_pass
    ;;
  status)
    if running; then
      echo "ON (pid $(cat "$PIDFILE"))"
    else
      echo "OFF"
    fi
    if [[ -f "$STATUSLOG" ]]; then echo "-- last passes --"; tail -n 5 "$STATUSLOG"; fi
    ;;
  *)
    usage
    ;;
esac
