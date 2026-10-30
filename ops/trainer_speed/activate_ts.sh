#!/usr/bin/env bash
# RA/20261030-trainer-speed activation: trainer v8.3.3f -> v8.3.3g (execution-only speedups; bit-identical update math, CPU-tested)
#   Jacob pt841u + t847u ("push in all the other optimization fixes ... go; I don't care if you interrupt").
#   MODE=switch   install now if the trainer is idle (no queued/leased train job, no RED restore running); starts ts-watchdog.
#   MODE=revert   v8.3.3g -> v8.3.3f (waits for the running job unless FORCE=1). One-line revert:
#                 MODE=revert bash /home/const/subnet120-rewrite/state/root-audits/20261030-trainer-speed/activation/activate_ts.sh
set -euo pipefail; umask 077
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
MODE=${MODE:?MODE=switch|revert}
RA=/home/const/subnet120-rewrite/state/root-audits; D=$RA/20261030-trainer-speed; AV=$D/activation; mkdir -p -m 700 $AV
REL=$D/release-v833g-signed; STATE=/home/const/subnet120-rewrite/state/mimo-mainnet-v1; E=$STATE/epochs
PY=/home/const/subnet120-rewrite/.venv/bin/python
UNITS=/home/const/.config/systemd/user; SUP=affine-mimo-trainer-supervised.service; TD=$UNITS/$SUP.d
F_TC='zzzz~~~~~~~~~~~~~~~v833f-owned-worker.conf'; G_TC='zzzz~~~~~~~~~~~~~~~~v833g-owned-worker.conf'
F_J=owned-worker-v833f.json; F_S=e8d6c843e05d0b0df7d59d07411fa789d04e411307524dd787acc3b14de16818; G_J=owned-worker-v833g.json
NEED_GB=192; POD=/workspace/affine-mimo-trainer-prod; OW="$POD/runtime-exact/venv/bin/python -I -B $POD/owned-worker-v7.py"
TKH=/home/const/.lium/known_hosts/ef2b6085-adae-43e2-8a13-39182a3b3f7a
SSHO=(-T -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10 -o ServerAliveInterval=20 -o ServerAliveCountMax=6)
N(){ local i rc; for i in 1 2 3 4 5 6; do ssh "${SSHO[@]}" -o UserKnownHostsFile=$TKH -p 40022 root@8.28.35.244 "$@" && return 0; rc=$?; [[ $rc == 255 ]] || return $rc; sleep 5; done; return 255; }
cr(){ TZ=America/Costa_Rica date "+%H:%M:%S CR"; }
LOG=$AV/activate.log; log(){ echo "$(date -u +%FT%TZ) $(cr) [$MODE] $*" | tee -a "$LOG"; }
prog(){ echo "- $(TZ=America/Costa_Rica date '+%H:%M CR'): activate_ts $MODE: $*" >> "$D/PROGRESS.md"; }
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
  N 'pids=""; for p in $(pgrep -f "[h]ost-v833[fg]/v83_dp.py"); do tr "\0" " " </proc/$p/cmdline | grep -q "host-v833[fg]/v83_dp.py" && pids="$pids $p"; done
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
  if [[ $act == install ]]; then
    local line new; line=$(grep '^ExecStart="/usr/bin/ssh"' "$TD/$F_TC"); new=${line//$F_J --sha256 $F_S/$G_J --sha256 $G_S}
    [[ "$new" != "$line" && "$new" == *"$G_J --sha256 $G_S"* ]] || return 15
    printf '%s\n%s\n%s\n%s\n%s\n' "# MiMo trainer v8.3.3g (= v8.3.3f + execution-only speedups) on golden-raven-7a. Jacob pt841u/t847u 2026-10-30." \
      "# Installed by $AV/activate_ts.sh $(cr). Revert: MODE=revert bash $AV/activate_ts.sh" "[Service]" "ExecStart=" "$new" > "$AV/$G_TC"
    install -m 600 "$AV/$G_TC" "$TD/$G_TC"
  else rm -f "$TD/$G_TC"; fi
  systemctl --user daemon-reload
  grep -q "$tj --sha256 $ts" <(systemctl --user show $SUP -p ExecStart --value) || return 16
  systemctl --user start $SUP
  ok=0; for i in $(seq 1 60); do N "$OW --config $POD/$tj --sha256 $ts --status" | grep -q '"alive": true' && { ok=1; break; }; sleep 5; done
  [[ $ok == 1 ]] || return 17
  log "trainer -> $tj ($ts) alive"; return 0; }
stage(){
  (cd $REL && sha256sum --quiet -c SIGNED-MANIFEST.sha256) || die "release manifest"
  tar cf - -C $REL trainer-host-v833g | N "set -e; cd $POD; if [ ! -d host-v833g ]; then rm -rf .host-v833g.new; mkdir -m 700 .host-v833g.new; tar --no-same-owner -xf - -C .host-v833g.new; chown -R root:root .host-v833g.new; mv .host-v833g.new/trainer-host-v833g host-v833g; rmdir .host-v833g.new; else cat >/dev/null; fi
    sed 's#/host-v833f/#/host-v833g/#g' $F_J > .$G_J.new; chmod 600 .$G_J.new; mv .$G_J.new $G_J"
  (cd $REL/trainer-host-v833g && find . -type f -print0 | sort -z | xargs -0 sha256sum) > $AV/host-v833g.sha256
  N "cd $POD/host-v833g && sha256sum --quiet -c -" < $AV/host-v833g.sha256 || die "pod host-v833g sha mismatch"
  N "cd $POD && diff <(sed 's#/host-v833g/#/host-v833f/#g' $G_J) $F_J >/dev/null && ! grep -q host-v833f $G_J" || die "worker config diff"
  N "cd $POD/source-final-v823u && env CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 MKL_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH=$POD/source-final-v823u TOKENIZERS_PARALLELISM=false timeout 240 $POD/runtime-exact/venv/bin/python -B $POD/host-v833g/mimo_host_trainer_v83.py --operator-config $POD/host-v833g/TRAINER-HOST.v8.3.3.ROOT-ADMITTED.private.json --check -- >/dev/null 2>&1" || die "v833g --check failed (admission/pins/install)"
  G_S=$(N "sha256sum < $POD/$G_J" | cut -d' ' -f1); [[ $G_S =~ ^[0-9a-f]{64}$ ]] || die "G_S"; echo "$G_S" > $AV/G_S; log "staged host-v833g + $G_J ($G_S)"; }
diskok(){ local f; f=$(N "df -B1G --output=avail $POD | tail -1" | tr -d ' '); [[ $f =~ ^[0-9]+$ ]] && (( f >= NEED_GB )) && { log "pod free ${f} GB >= $NEED_GB"; return 0; }; log "pod free ${f:-?} GB < $NEED_GB"; return 1; }
case $MODE in
switch)
  [[ -f $TD/$G_TC ]] && die "v833g drop-in already installed"
  [[ -n ${AFTER_OK:-} ]] && { log "waiting for $AFTER_OK"; while ! ls $AFTER_OK >/dev/null 2>&1; do sleep 10; done; log "found $(ls $AFTER_OK | head -1)"; }
  grep -q "$F_J --sha256 $F_S" <(systemctl --user show $SUP -p ExecStart --value) || die "live trainer is not v833f (rebase needed)"
  diskok || die "pod disk precheck failed"
  stage
  B=$AV/backup-$(date -u +%Y%m%dT%H%M%SZ); mkdir -m 700 $B; cp -a $TD $B/dropins; systemctl --user show $SUP -p ExecStart --value > $B/ExecStart.txt
  if [[ -n ${WAIT_DECISION:-} ]]; then GE=$RA/20261026-promotion-gate/state/epochs
    log "waiting for gate DECISION e$WAIT_DECISION (+ RED restore) before switching"
    while [[ ! -f $GE/$WAIT_DECISION/DECISION.json ]]; do sleep 1; done
    if grep -q '"decision": *"red"' $GE/$WAIT_DECISION/DECISION.json; then
      for i in $(seq 1 450); do [[ -f $GE/$WAIT_DECISION/TRAINER-RESTORED.json ]] && break; sleep 2; done; sleep 3; fi
  fi
  ok=0; for i in $(seq 1 ${IWAIT:-720}); do
    if idle; then exec 8<>$STATE/lifecycle.lock; exec 9>$RA/LIFECYCLE-ACTIVATION.lock
      if flock -w 30 9 && flock -w 30 8; then systemctl --user stop affine-mimo-lifecycle.service; c=$(cur)
        if ! openjob $c && [[ $(trainbusy) == 0 ]] && diskok; then ok=1; break; fi
        systemctl --user start affine-mimo-lifecycle.service; fi; exec 8>&- 9>&-; fi
    sleep 5; done
  [[ $ok == 1 ]] || die "trainer never idle"
  log "trainer idle (epoch $(cur)), lifecycle stopped + lock held - switching v833f -> v833g"
  systemctl --user stop ts-watchdog-f.service 2>/dev/null || true; log "ts-watchdog-f stopped (v833f watch ends at the v833g switch)"
  rc=0; switch $F_J $F_S $G_J $G_S install || rc=$?
  if [[ $rc != 0 ]]; then log "v833g switch failed rc=$rc - restoring v833f"; rm -f $TD/$G_TC; systemctl --user daemon-reload; switch $G_J $G_S $F_J $F_S remove || log "WARNING restore rc=$?"; systemctl --user start affine-mimo-lifecycle.service; die "switch failed rc=$rc (v833f restored)"; fi
  systemctl --user start affine-mimo-lifecycle.service; exec 8>&- 9>&-; log "lifecycle started, locks released"
  echo "{\"trainer\": \"v833g\", \"sha256\": \"$G_S\", \"from_epoch\": $(cur), \"at\": \"$(cr)\"}" > $AV/TRAINER-g.json
  W=$(cur); while [[ -e $E/$W/TRAINING-JOB.json ]]; do W=$((W+1)); done   # first epoch trained by v833g
  systemctl --user reset-failed ts-watchdog.service 2>/dev/null || true
  systemd-run --user --unit=ts-watchdog --collect -p Restart=on-failure --setenv=FROM=$W --setenv=LIMIT=${LIMIT:-960} $PY -u -B $D/release/watchdog_ts.py
  prog "LIVE v8.3.3g (worker cfg $G_S) at epoch $(cur); ts-watchdog started"; log done ;;
revert)
  G_S=$(cat $AV/G_S); [[ -f $TD/$G_TC ]] || die "v833g drop-in not installed"
  [[ ${FROM_WATCHDOG:-0} == 1 ]] || systemctl --user stop ts-watchdog.service 2>/dev/null || true
  if [[ ${FORCE:-0} != 1 ]]; then for i in $(seq 1 ${IWAIT:-720}); do idle && break; sleep 5; done; fi
  exec 9>$RA/LIFECYCLE-ACTIVATION.lock; flock -w 120 9 || true; exec 8<>$STATE/lifecycle.lock; flock -w 60 8 || true; systemctl --user stop affine-mimo-lifecycle.service
  rc=0; switch $G_J $G_S $F_J $F_S remove || rc=$?
  systemctl --user start affine-mimo-lifecycle.service; exec 8>&- 9>&-; [[ $rc == 0 ]] || die "revert failed rc=$rc"
  rm -f $AV/TRAINER-g.json; prog "REVERTED to v8.3.3f (${REASON:-manual})"; log done ;;
esac
