#!/usr/bin/env bash
# RA/20261030-trainer-speed: re-activate trainer v8.3.3f (Allan fp32-master) after the 14:13 CR false-positive fp32v revert (parent 14:16 CR).
#   MODE=switch   v8.3.3u -> v8.3.3f at the first safe boundary (fixed idle logic + lifecycle & activation locks + pod disk precheck);
#                 installs the sanctioned fp32v drop-in (RA/20261030-fp32-verifier/activation/zzzz~~~~~~~~~~~~~~~v833f-owned-worker.conf)
#                 and starts ts-watchdog-f (trainer-caused faults only; disk-check failures never counted).
#   MODE=revert   v8.3.3f -> v8.3.3u:  MODE=revert bash /home/const/subnet120-rewrite/state/root-audits/20261030-trainer-speed/activation/activate_f.sh
set -euo pipefail; umask 077
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
MODE=${MODE:?MODE=switch|revert}
RA=/home/const/subnet120-rewrite/state/root-audits; D=$RA/20261030-trainer-speed; AV=$D/activation; mkdir -p -m 700 $AV
REL=$D/release-v833g-signed; STATE=/home/const/subnet120-rewrite/state/mimo-mainnet-v1; E=$STATE/epochs
PY=/home/const/subnet120-rewrite/.venv/bin/python
UNITS=/home/const/.config/systemd/user; SUP=affine-mimo-trainer-supervised.service; TD=$UNITS/$SUP.d
FV=$RA/20261030-fp32-verifier/activation; F_TC='zzzz~~~~~~~~~~~~~~~v833f-owned-worker.conf'
U_J=owned-worker-v833u.json; U_S=c7a644df8f0efc2dc172fecb9479904fd879c41e276ca78cbdb3b8d17cdb39b6
F_J=owned-worker-v833f.json; F_S=e8d6c843e05d0b0df7d59d07411fa789d04e411307524dd787acc3b14de16818
NEED_GB=192   # 135.1 GB job disk check + 36 GB fp32 entry + 20 GB margin
POD=/workspace/affine-mimo-trainer-prod; OW="$POD/runtime-exact/venv/bin/python -I -B $POD/owned-worker-v7.py"
TKH=/home/const/.lium/known_hosts/ef2b6085-adae-43e2-8a13-39182a3b3f7a
SSHO=(-T -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10 -o ServerAliveInterval=20 -o ServerAliveCountMax=6)
N(){ local i rc; for i in 1 2 3 4 5 6; do ssh "${SSHO[@]}" -o UserKnownHostsFile=$TKH -p 40022 root@8.28.35.244 "$@" && return 0; rc=$?; [[ $rc == 255 ]] || return $rc; sleep 5; done; return 255; }
cr(){ TZ=America/Costa_Rica date "+%H:%M:%S CR"; }
LOG=$AV/activate_f.log; log(){ echo "$(date -u +%FT%TZ) $(cr) [$MODE] $*" | tee -a "$LOG"; }
prog(){ echo "- $(TZ=America/Costa_Rica date '+%H:%M CR'): activate_f $MODE: $*" >> "$D/PROGRESS.md"; }
die(){ log "ABORT: $*"; prog "ABORT: $*"; exit 1; }
trainbusy(){ timeout 12 $PY -c "
import sqlite3;c=sqlite3.connect('file:$STATE/roles.sqlite3?mode=ro',uri=True,timeout=5)
print(c.execute(\"select count(*) from jobs where role='train' and status not in ('complete','failed','expired')\").fetchone()[0])" 2>/dev/null || echo err; }
cur(){ $PY -c "import json;print(json.load(open('$STATE/CURRENT.json'))['epoch_index'])"; }
openjob(){ # CURRENT lags: the lifecycle pre-opens/dispatches e(C+1) while CURRENT=C (V829V-PREOPEN) -> check C..C+2
  local x; for x in $1 $(($1+1)) $(($1+2)); do
    [[ -e $E/$x/RECOVERY-REQUIRED.json ]] && return 0
    [[ -e $E/$x/TRAINING-JOB.json && ! -e $E/$x/TRAINING-RESULT.json ]] && return 0; done; return 1; }
idle(){ # 0 = trainer provably idle: lifecycle active (no other operator), current epoch has no open TRAINING-JOB / RECOVERY-REQUIRED,
  # no train row outside complete/failed/expired, no gate restore, no fp32 master save in flight.
  # (13:21 CR incident: e276 was dispatched while CURRENT=275 and I only looked at CURRENT -> check C..C+2 + DB.)
  local c; c=$(cur)
  systemctl --user is-active --quiet affine-mimo-lifecycle.service || return 1
  openjob $c && return 1
  systemctl --user list-units --no-legend 'affine-gate-restore-*' | grep -q -E 'active|activating' && return 1
  [[ $(trainbusy) == 0 ]] || return 1
  N "find $POD/fp32-master -newermt '-75 seconds' | grep -q ." && return 1
  return 0; }
kill_helpers(){ # PID-scoped: list v83_dp helper PIDs, verify each cmdline, TERM then KILL those PIDs only
  N 'pids=""; for p in $(pgrep -f "[h]ost-v833[a-z]*/v83_dp.py"); do tr "\0" " " </proc/$p/cmdline | grep -q "host-v833[a-z]*/v83_dp.py" && pids="$pids $p"; done
     echo "helpers:$pids"; [ -z "$pids" ] || { kill -TERM $pids 2>/dev/null; sleep 5; for p in $pids; do [ -d /proc/$p ] && kill -KILL $p 2>/dev/null; done; true; }' | tee -a "$LOG"; }
switch(){ # FROM_J FROM_S TO_J TO_S install|remove
  local fj=$1 fs=$2 tj=$3 ts=$4 act=$5 i ok
  systemctl --user stop $SUP
  N "$OW --config $POD/$fj --sha256 $fs --stop-owned-after-drain" 2>&1 | cut -c1-300 | tee -a "$LOG" || true
  ok=0; for i in $(seq 1 60); do N "$OW --config $POD/$fj --sha256 $fs --status" | grep -q '"alive": false' && { ok=1; break; }; sleep 5; done
  [[ $ok == 1 ]] || return 11
  kill_helpers
  for i in $(seq 1 12); do N "ss -ltn | grep -q ':29583 '" || break; sleep 5; done
  N "ss -ltn | grep -q ':29583 '" && return 12
  if [[ $act == install ]]; then grep -q "$F_J --sha256 $F_S" "$FV/$F_TC" || return 15; install -m 600 "$FV/$F_TC" "$TD/$F_TC"
  else rm -f "$TD/$F_TC"; fi
  systemctl --user daemon-reload
  grep -q "$tj --sha256 $ts" <(systemctl --user show $SUP -p ExecStart --value) || return 16
  systemctl --user start $SUP
  ok=0; for i in $(seq 1 60); do N "$OW --config $POD/$tj --sha256 $ts --status" | grep -q '"alive": true' && { ok=1; break; }; sleep 5; done
  [[ $ok == 1 ]] || return 17
  log "trainer -> $tj ($ts) alive"; return 0; }
diskok(){ local f; f=$(N "df -B1G --output=avail $POD | tail -1" | tr -d ' '); [[ $f =~ ^[0-9]+$ ]] && (( f >= NEED_GB )) && { log "pod free ${f} GB >= $NEED_GB"; return 0; }; log "pod free ${f:-?} GB < $NEED_GB"; return 1; }
case $MODE in
switch)
  [[ -f $TD/$F_TC ]] && die "v833f drop-in already installed"
  grep -q "$U_J --sha256 $U_S" <(systemctl --user show $SUP -p ExecStart --value) || die "live trainer is not v833u"
  (cd $RA/20261030-fp32-verifier/release-v833f-signed && sha256sum --quiet -c SIGNED-MANIFEST.sha256) || die "v833f release manifest"
  [[ $(N "sha256sum < $POD/$F_J" | cut -d' ' -f1) == $F_S ]] || die "v833f worker config not on pod"
  [[ $(N "sha256sum < $POD/host-v833f/mimo_host_trainer_v83.py" | cut -d' ' -f1) == dde1eeb1d510b3dbad7963f8d68190bd6fad8a5a3187bb5d694c74143dcb34cf ]] || die "v833f launcher not on pod"
  diskok || die "pod disk precheck failed"
  B=$AV/backup-f-$(date -u +%Y%m%dT%H%M%SZ); mkdir -m 700 $B; cp -a $TD $B/dropins; systemctl --user show $SUP -p ExecStart --value > $B/ExecStart.txt
  ok=0; for i in $(seq 1 ${IWAIT:-1440}); do
    if idle; then exec 8<>$STATE/lifecycle.lock; exec 9>$RA/LIFECYCLE-ACTIVATION.lock
      if flock -w 30 9 && flock -w 30 8; then systemctl --user stop affine-mimo-lifecycle.service; c=$(cur)
        if ! openjob $c && [[ $(trainbusy) == 0 ]] && diskok; then ok=1; break; fi
        systemctl --user start affine-mimo-lifecycle.service; fi; exec 8>&- 9>&-; fi
    sleep 5; done
  [[ $ok == 1 ]] || die "trainer never idle"
  log "boundary (CURRENT $(cur)): lifecycle stopped, locks held - switching v833u -> v833f"
  rc=0; switch $U_J $U_S $F_J $F_S install || rc=$?
  if [[ $rc != 0 ]]; then log "v833f switch failed rc=$rc - restoring v833u"; rm -f $TD/$F_TC; systemctl --user daemon-reload; switch $F_J $F_S $U_J $U_S remove || log "WARNING restore rc=$?"; systemctl --user start affine-mimo-lifecycle.service; die "switch failed rc=$rc (v833u restored)"; fi
  systemctl --user start affine-mimo-lifecycle.service; exec 8>&- 9>&-; log "lifecycle started, locks released"
  W=$(cur); while [[ -e $E/$W/TRAINING-JOB.json ]]; do W=$((W+1)); done
  echo "{\"trainer\": \"v833f\", \"from_epoch\": $W, \"at\": \"$(cr)\", \"by\": \"trainer-speed activate_f\"}" > $AV/TRAINER-f.json
  systemctl --user reset-failed ts-watchdog-f.service 2>/dev/null || true
  systemd-run --user --unit=ts-watchdog-f --collect -p Restart=on-failure --setenv=FROM=$W --setenv=CFG=$F_J --setenv=CFG_S=$F_S --setenv=REVERT=$AV/activate_f.sh --setenv=WATCH=${WATCH:-2} $PY -u -B $D/release/watchdog_ts.py
  prog "LIVE v8.3.3f (re-activated after the false-positive fp32v revert) from e$W; ts-watchdog-f armed"; log done ;;
revert)
  [[ -f $TD/$F_TC ]] || die "v833f drop-in not installed"
  [[ ${FROM_WATCHDOG:-0} == 1 ]] || systemctl --user stop ts-watchdog-f.service 2>/dev/null || true
  if [[ ${FORCE:-0} != 1 ]]; then for i in $(seq 1 ${IWAIT:-720}); do idle && break; sleep 5; done; fi
  exec 9>$RA/LIFECYCLE-ACTIVATION.lock; flock -w 120 9 || true; exec 8<>$STATE/lifecycle.lock; flock -w 60 8 || true; systemctl --user stop affine-mimo-lifecycle.service
  rc=0; switch $F_J $F_S $U_J $U_S remove || rc=$?
  systemctl --user start affine-mimo-lifecycle.service; exec 8>&- 9>&-; [[ $rc == 0 ]] || die "revert failed rc=$rc"
  prog "REVERTED v8.3.3f -> v8.3.3u (${REASON:-manual})"; log done ;;
esac
