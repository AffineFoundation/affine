#!/usr/bin/env bash
# RA/20261031-renorm-step-live activation: trainer v8.3.3h -> v8.3.3i (fp32 master kept + normgrad direction step + support renorm).
#   MODE=stage   verify + stage host-v833i and owned-worker-v833i.json on the trainer pod (+ --check). Inert.
#   MODE=switch  waits for the gate DECISION of the trained epoch and an idle trainer; with the lifecycle stopped and both locks held:
#                trainer drop-in v833h -> v833i. Lifecycle (ff_launcher), gate, batch cap, queue, free-file rules untouched.
#   MODE=revert  v833i -> v833h (waits for an idle trainer unless FORCE=1). One-line revert:
#                MODE=revert bash /home/const/subnet120-rewrite/state/root-audits/20261031-renorm-step-live/activation/activate_rs.sh
set -euo pipefail; umask 077
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
MODE=${MODE:?MODE=stage|switch|revert}
RA=/home/const/subnet120-rewrite/state/root-audits; D=$RA/20261031-renorm-step-live; AV=$D/activation; mkdir -p -m 700 $AV
REL=$D/release-v833i-signed; STATE=/home/const/subnet120-rewrite/state/mimo-mainnet-v1; E=$STATE/epochs
GE=$RA/20261026-promotion-gate/state; PY=/home/const/subnet120-rewrite/.venv/bin/python
UNITS=/home/const/.config/systemd/user; SUP=affine-mimo-trainer-supervised.service; TD=$UNITS/$SUP.d
H_TC='zzzz~~~~~~~~~~~~~~~~~v833h-owned-worker.conf'; I_TC='zzzz~~~~~~~~~~~~~~~~~~v833i-owned-worker.conf'
H_J=owned-worker-v833h.json; H_S=12b01d395d23d7cf023eb1fc1d844fe79a0cce3c9a44530187634aa978478983; I_J=owned-worker-v833i.json
NEED_GB=192; POD=/workspace/affine-mimo-trainer-prod; OW="$POD/runtime-exact/venv/bin/python -I -B $POD/owned-worker-v7.py"
TKH=/home/const/.lium/known_hosts/ef2b6085-adae-43e2-8a13-39182a3b3f7a
SSHO=(-T -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10 -o ServerAliveInterval=20 -o ServerAliveCountMax=6)
N(){ local i rc; for i in 1 2 3 4 5 6; do ssh "${SSHO[@]}" -o UserKnownHostsFile=$TKH -p 40022 root@8.28.35.244 "$@" && return 0; rc=$?; [[ $rc == 255 ]] || return $rc; sleep 5; done; return 255; }
cr(){ TZ=America/Costa_Rica date "+%H:%M:%S CR"; }
LOG=$AV/activate.log; log(){ echo "$(date -u +%FT%TZ) $(cr) [$MODE] $*" | tee -a "$LOG"; }
prog(){ echo "- $(TZ=America/Costa_Rica date '+%H:%M CR'): activate_rs $MODE: $*" >> "$D/PROGRESS.md"; }
die(){ log "ABORT: $*"; prog "ABORT: $*"; exit 1; }
trainbusy(){ timeout 12 $PY -c "
import sqlite3;c=sqlite3.connect('file:$STATE/roles.sqlite3?mode=ro',uri=True,timeout=5)
print(c.execute(\"select count(*) from jobs where role='train' and status not in ('complete','failed','expired')\").fetchone()[0])" 2>/dev/null || echo err; }
cur(){ $PY -c "import json;print(json.load(open('$STATE/CURRENT.json'))['epoch_index'])"; }
openjob(){ local x; for x in $1 $(($1+1)) $(($1+2)); do
    [[ -e $E/$x/RECOVERY-REQUIRED.json ]] && return 0
    [[ -e $E/$x/TRAINING-JOB.json && ! -e $E/$x/TRAINING-RESULT.json ]] && return 0; done; return 1; }
idle(){ local c; c=$(cur)
  systemctl --user is-active --quiet affine-mimo-lifecycle.service || return 1
  openjob $c && return 1
  local x; for x in $((c-1)) $c $((c+1)) $((c+2)); do
    [[ -e $E/$x/TRAINING-RESULT.json && ! -e $GE/epochs/$x/DECISION.json ]] && return 1
    [[ -e $GE/epochs/$x/DECISION.json ]] && grep -q '"decision": *"red"' $GE/epochs/$x/DECISION.json && [[ ! -e $GE/epochs/$x/TRAINER-RESTORED.json ]] && return 1
  done
  systemctl --user list-units --no-legend 'affine-gate-restore-*' | grep -q -E 'active|activating' && return 1
  [[ $(trainbusy) == 0 ]] || return 1
  N "find $POD/fp32-master -newermt '-75 seconds' | grep -q ." && return 1
  return 0; }
kill_helpers(){ # PID-scoped: v83_dp helper PIDs (cmdline verified), TERM then KILL those exact PIDs only
  N 'pids=""; for p in $(pgrep -f "[h]ost-v833[hi]/v83_dp.py"); do tr "\0" " " </proc/$p/cmdline | grep -q "host-v833[hi]/v83_dp.py" && pids="$pids $p"; done
     echo "helpers:$pids"; [ -z "$pids" ] || { kill -TERM $pids 2>/dev/null; sleep 5; for p in $pids; do [ -d /proc/$p ] && kill -KILL $p 2>/dev/null; done; true; }' | tee -a "$LOG"; }
switch_trainer(){ # FROM_J FROM_S TO_J TO_S install|remove
  local fj=$1 fs=$2 tj=$3 ts=$4 act=$5 i ok
  systemctl --user stop $SUP
  N "$OW --config $POD/$fj --sha256 $fs --stop-owned-after-drain" 2>&1 | cut -c1-300 | tee -a "$LOG" || true
  ok=0; for i in $(seq 1 60); do N "$OW --config $POD/$fj --sha256 $fs --status" | grep -q '"alive": false' && { ok=1; break; }; sleep 5; done
  [[ $ok == 1 ]] || return 11
  kill_helpers
  for i in $(seq 1 12); do N "ss -ltn | grep -q ':29583 '" || break; sleep 5; done
  N "ss -ltn | grep -q ':29583 '" && return 12
  if [[ $act == install ]]; then
    local line new; line=$(grep '^ExecStart="/usr/bin/ssh"' "$TD/$H_TC"); new=${line//$H_J --sha256 $H_S/$I_J --sha256 $I_S}
    [[ "$new" != "$line" && "$new" == *"$I_J --sha256 $I_S"* ]] || return 15
    printf '%s\n%s\n%s\n%s\n%s\n' "# MiMo trainer v8.3.3i (= v8.3.3h + normgrad direction step + support renorm). Jacob 2026-10-31 07:00 CR." \
      "# Installed by $AV/activate_rs.sh $(cr). Revert: MODE=revert bash $AV/activate_rs.sh" "[Service]" "ExecStart=" "$new" > "$AV/$I_TC"
    install -m 600 "$AV/$I_TC" "$TD/$I_TC"
  else rm -f "$TD/$I_TC"; fi
  systemctl --user daemon-reload
  grep -q "$tj --sha256 $ts" <(systemctl --user show $SUP -p ExecStart --value) || return 16
  systemctl --user start $SUP
  ok=0; for i in $(seq 1 60); do N "$OW --config $POD/$tj --sha256 $ts --status" | grep -q '"alive": true' && { ok=1; break; }; sleep 5; done
  [[ $ok == 1 ]] || return 17
  log "trainer -> $tj ($ts) alive"; return 0; }
diskok(){ local f; f=$(N "df -B1G --output=avail $POD | tail -1" | tr -d ' '); [[ $f =~ ^[0-9]+$ ]] && (( f >= NEED_GB )) && { log "pod free ${f} GB >= $NEED_GB"; return 0; }; log "pod free ${f:-?} GB < $NEED_GB"; return 1; }
stage(){
  (cd $REL && sha256sum --quiet -c SIGNED-MANIFEST.sha256) || die "release manifest"
  N "df -T $POD | tail -1" | grep -qi fuse && die "pod path on fuse (gocryptfs) - refusing"
  tar cf - -C $REL trainer-host-v833i | N "set -e; cd $POD; if [ ! -d host-v833i ]; then rm -rf .host-v833i.new; mkdir -m 700 .host-v833i.new; tar --no-same-owner -xf - -C .host-v833i.new; chown -R root:root .host-v833i.new; mv .host-v833i.new/trainer-host-v833i host-v833i; rmdir .host-v833i.new; else cat >/dev/null; fi
    sed 's#/host-v833h/#/host-v833i/#g' $H_J > .$I_J.new; chmod 600 .$I_J.new; mv .$I_J.new $I_J"
  (cd $REL/trainer-host-v833i && find . -type f -print0 | sort -z | xargs -0 sha256sum) > $AV/host-v833i.sha256
  N "cd $POD/host-v833i && sha256sum --quiet -c -" < $AV/host-v833i.sha256 || die "pod host-v833i sha mismatch"
  N "cd $POD && diff <(sed 's#/host-v833i/#/host-v833h/#g' $I_J) $H_J >/dev/null && ! grep -q host-v833h $I_J" || die "worker config diff"
  N "cd $POD/source-final-v823u && env CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 MKL_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH=$POD/source-final-v823u TOKENIZERS_PARALLELISM=false timeout 240 $POD/runtime-exact/venv/bin/python -B $POD/host-v833i/mimo_host_trainer_v83.py --operator-config $POD/host-v833i/TRAINER-HOST.v8.3.3.ROOT-ADMITTED.private.json --check -- > /tmp/v833i-check.out 2>&1; rc=\$?; tail -c 1500 /tmp/v833i-check.out; exit \$rc" > $AV/check.out 2>&1 || die "v833i --check failed (admission/pins/install): $(tail -c 400 $AV/check.out)"
  I_S=$(N "sha256sum < $POD/$I_J" | cut -d' ' -f1); [[ $I_S =~ ^[0-9a-f]{64}$ ]] || die "I_S"; echo "$I_S" > $AV/I_S
  log "staged host-v833i + $I_J ($I_S)"; }
case $MODE in
stage) stage; prog "staged (trainer v833i on pod, --check OK, worker config $(cat $AV/I_S))"; log done ;;
switch)
  [[ -f $TD/$I_TC ]] && die "v833i drop-in already installed"
  grep -q "$H_J --sha256 $H_S" <(systemctl --user show $SUP -p ExecStart --value) || die "live trainer is not v833h"
  [[ $(ls $TD | tail -1) == "$H_TC" ]] || die "unexpected newer trainer drop-in"
  diskok || die "pod disk precheck failed"
  stage; I_S=$(cat $AV/I_S)
  B=$AV/backup-$(date -u +%Y%m%dT%H%M%SZ); mkdir -m 700 $B; echo $B > $AV/LAST-BACKUP
  cp -a $TD $B/trainer-dropins; systemctl --user show $SUP -p ExecStart --value > $B/trainer-ExecStart.txt
  X=$(( $(cur) - 1 )); for x in $X $((X+1)) $((X+2)) $((X+3)); do [[ -e $E/$x/TRAINING-RESULT.json ]] && X=$x; done
  if [[ ! -f $GE/epochs/$X/DECISION.json ]]; then log "waiting for gate DECISION e$X (+ RED restore) before switching"
    while [[ ! -f $GE/epochs/$X/DECISION.json ]]; do sleep 1; done
    if grep -q '"decision": *"red"' $GE/epochs/$X/DECISION.json; then
      for i in $(seq 1 450); do [[ -f $GE/epochs/$X/TRAINER-RESTORED.json ]] && break; sleep 2; done; sleep 3; fi
  fi
  ok=0; for i in $(seq 1 ${IWAIT:-720}); do
    if idle; then exec 8<>$STATE/lifecycle.lock; exec 9>$RA/LIFECYCLE-ACTIVATION.lock
      if flock -w 30 9 && flock -w 30 8; then systemctl --user stop affine-mimo-lifecycle.service; c=$(cur)
        if ! openjob $c && [[ $(trainbusy) == 0 ]] && diskok; then ok=1; break; fi
        systemctl --user start affine-mimo-lifecycle.service; fi; exec 8>&- 9>&-; fi
    sleep 5; done
  [[ $ok == 1 ]] || die "trainer never idle"
  log "trainer idle (epoch $(cur), last decided e$X), lifecycle stopped + locks held - switching"
  rc=0; switch_trainer $H_J $H_S $I_J $I_S install || rc=$?
  if [[ $rc != 0 ]]; then log "v833i switch failed rc=$rc - restoring v833h"; rm -f $TD/$I_TC; systemctl --user daemon-reload
    switch_trainer $I_J $I_S $H_J $H_S remove || log "WARNING restore rc=$?"; systemctl --user start affine-mimo-lifecycle.service; die "trainer switch failed rc=$rc (v833h restored)"; fi
  W=$(cur); while [[ -e $E/$W/TRAINING-JOB.json ]]; do W=$((W+1)); done      # first epoch whose job the v833i trainer runs
  systemctl --user start affine-mimo-lifecycle.service; exec 8>&- 9>&-; log "lifecycle started, locks released"
  echo "{\"from_epoch\": $W, \"trainer\": \"v833i\", \"sha256\": \"$I_S\", \"at\": \"$(cr)\", \"backup\": \"$B\", \"last_v833h_decided_epoch\": $X}" > $AV/ACTIVE.json
  prog "LIVE trainer v833i from epoch $W ($I_S); last v833h turn decided: e$X"; log done ;;
revert)
  [[ -f $TD/$I_TC ]] || { log "v833i drop-in not installed - nothing to revert"; rm -f $AV/ACTIVE.json; exit 0; }
  I_S=$(cat $AV/I_S)
  if [[ ${FORCE:-0} != 1 ]]; then for i in $(seq 1 ${IWAIT:-720}); do idle && break; sleep 5; done; fi
  exec 9>$RA/LIFECYCLE-ACTIVATION.lock; flock -w 120 9 || true; exec 8<>$STATE/lifecycle.lock; flock -w 60 8 || true
  systemctl --user stop affine-mimo-lifecycle.service
  rc=0; switch_trainer $I_J $I_S $H_J $H_S remove || rc=$?
  systemctl --user start affine-mimo-lifecycle.service; exec 8>&- 9>&-
  [[ -f $AV/ACTIVE.json ]] && mv -f $AV/ACTIVE.json $AV/REVERTED-$(date -u +%Y%m%dT%H%M%SZ).json
  [[ $rc == 0 ]] || die "trainer revert failed rc=$rc (lifecycle restarted)"
  prog "REVERTED trainer v833i -> v833h (${REASON:-manual})"; log done ;;
esac
