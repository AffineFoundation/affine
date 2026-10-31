#!/usr/bin/env bash
# Step 1 runner (RA/20261031-legacy-removal): wait for epoch WAIT_FIN's FINALIZED (gate DECISION), then, holding
# LIFECYCLE-ACTIVATION.lock (shared with the other activation jobs), run legacy_off.sh MODE=apply and arm legacy-watchdog
# FROM=WAIT_FIN+1. If the lock is held by another job, wait for the next DECISION gap. Revert: MODE=revert bash legacy_off.sh
set -uo pipefail
RA=/home/const/subnet120-rewrite/state/root-audits; D=$RA/20261031-legacy-removal; OFF=$RA/20261030-free-file/activation/legacy_off.sh
E=/home/const/subnet120-rewrite/state/mimo-mainnet-v1/epochs; PY=/home/const/subnet120-rewrite/.venv/bin/python
n=${WAIT_FIN:?}; cr(){ TZ=America/Costa_Rica date "+%H:%M:%S CR"; }
log(){ echo "- $(cr | cut -c1-5) CR: step1: $*" >> $D/PROGRESS.md; echo "$(cr) $*" >> $D/step1.log; }
log "waiting for e$n FINALIZED (DECISION gap)"
while true; do
  while [ ! -e $E/$n/FINALIZED.json ]; do sleep 2; done
  if [ -e $E/$((n+1))/TRAINING-JOB.json ]; then log "e$((n+1)) job already started; next gap"; n=$((n+1)); continue; fi
  exec 9>$RA/LIFECYCLE-ACTIVATION.lock
  if flock -n 9; then
    MODE=check bash $OFF >> $D/step1.log 2>&1 || { log "check NOT OK; aborting (nothing applied)"; exit 1; }
    MODE=apply bash $OFF >> $D/step1.log 2>&1; rc=$?
    exec 9>&-
    if [ $rc -ne 0 ]; then log "apply rc=$rc; reverting"; MODE=revert bash $OFF >> $D/step1.log 2>&1; exit 1; fi
    log "APPLIED after e$n DECISION ($(cat $RA/20261030-free-file/activation/LEGACY-OFF-LAST-BACKUP)); first epoch on flags: e$((n+1))"
    systemd-run --user --unit legacy-watchdog -p Restart=on-failure --setenv=FROM=$((n+1)) --setenv=N=3 $PY -u -B $D/watchdog_legacy.py
    log "legacy-watchdog armed FROM=e$((n+1)) N=3"; exit 0
  fi
  exec 9>&-; log "LIFECYCLE-ACTIVATION.lock held by another job after e$n; waiting for next gap"; n=$((n+1))
done
