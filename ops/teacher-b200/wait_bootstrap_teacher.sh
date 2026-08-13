#!/usr/bin/env bash
# Poll until affine-teacher is rented, then bootstrap GLM-Air.
set -euo pipefail
ROOT=/home/const/subnet120
EXP="$ROOT/ops/teacher-b200"
LOG="$EXP/logs/wait_bootstrap_teacher.log"
PIDF="$EXP/logs/wait_bootstrap_teacher.pid"
mkdir -p "$EXP/logs" "$EXP/artifacts"
echo $$ >"$PIDF"
exec >>"$LOG" 2>&1

# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
log() { echo "[teacher-boot-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

log "start"
for i in $(seq 1 20000); do
  if lium ps --format json | python3 -c '
import json,sys
pods=json.load(sys.stdin)
pods=pods if isinstance(pods,list) else pods.get("pods") or []
sys.exit(0 if any((p.get("name") or p.get("pod_name"))=="affine-teacher" for p in pods) else 1)
'; then
    log "affine-teacher live — bootstrap"
    # SSH may lag a few minutes after rent
    for j in $(seq 1 60); do
      if bash "$EXP/bootstrap_teacher.sh"; then
        log "BOOTSTRAP_OK"
        exit 0
      fi
      log "bootstrap attempt $j failed — retry in 30s"
      sleep 30
    done
    log "BOOTSTRAP_FAIL"
    exit 1
  fi
  if (( i == 1 || i % 30 == 0 )); then
    log "iter=$i waiting for rent stamp/pod"
  fi
  sleep 10
done
log "timeout"
exit 1
