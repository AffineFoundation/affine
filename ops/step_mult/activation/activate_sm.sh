#!/usr/bin/env bash
# RA/20261102-step-mult activation: trainer v8.3.3l = optional train.step_mult (integer in SM_ALLOWED, default 1) + the optional step / loss knobs
# in affine-batch-v2 free files, all defaulting to today's trainer (duongna's #120 proposal; Jacob 2026-11-02 10:19 CR "lets do this exactly
# like dou suggests and add a new hparam to the submission format"; 11:19 CR "ship all of these as optional knobs that start with the defaults").
# SM_ALLOWED (accepted step_mult values, e.g. 1,2,4; Jacob 11:48 CR) is read from $D/SM_ALLOWED (missing -> 1).
#   MODE=stage   verify + stage host-v833l and owned-worker-v833l.json on the trainer pod (+ --check); build host files in activation/next. Inert.
#   MODE=switch  at a ROUND BOUNDARY (the last decided turn is GREEN, i.e. a new held-out round just opened), in an idle decision gap no
#                other activation used, with the lifecycle stopped and both locks held: trainer v833j -> v833l, FF-PINS re-pinned, then
#                ff_format / ff_lifecycle / ff_service / affine_batch (+ website copy), FF-PUBLIC.json, training_params.py installed;
#                sm-watchdog armed (auto-revert on trainer trouble or holdout regression of a step_mult>1 GREEN king). The z>1 gate is untouched.
#   MODE=revert  (idle unless FORCE=1) everything back: trainer v833l -> v833j, ff files + pins, website client, FF-PUBLIC, training_params.
#                affine.io docs (code/web_docs_sm.py) restored too; GitHub is reverted separately (git revert). One-line revert:
#                MODE=revert bash /home/const/subnet120-rewrite/state/root-audits/20261102-step-mult/activation/activate_sm.sh
set -euo pipefail; umask 077
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
MODE=${MODE:?MODE=stage|switch|revert}
SM_ALLOWED=$(cat /home/const/subnet120-rewrite/state/root-audits/20261102-step-mult/SM_ALLOWED 2>/dev/null || echo 1); [[ $SM_ALLOWED =~ ^1(,[2-8])*$ ]] || SM_ALLOWED=1; export SM_ALLOWED
RA=/home/const/subnet120-rewrite/state/root-audits; D=$RA/20261102-step-mult; AV=$D/activation; mkdir -p -m 700 $AV
REL=$D/release-v833l-signed; STATE=/home/const/subnet120-rewrite/state/mimo-mainnet-v1; E=$STATE/epochs
GE=$RA/20261026-promotion-gate/state; PY=/home/const/subnet120-rewrite/.venv/bin/python
TPD=$RA/20261031-ff-train-params; FFD=$RA/20261030-free-file/release; WEBFF=/home/const/subnet120/affine/website/ff
TPC=$RA/20261026-promotion-gate/code/training_params.py; STG=$D/stage/release-l; CODE=$D/code2
UNITS=/home/const/.config/systemd/user; SUP=affine-mimo-trainer-supervised.service; TD=$UNITS/$SUP.d
J_TC='zzzz~~~~~~~~~~~~~~~~~~~v833j-owned-worker.conf'; K_TC='zzzz~~~~~~~~~~~~~~~~~~~~v833l-owned-worker.conf'
J_J=owned-worker-v833j.json; J_S=$(cat $TPD/activation/J_S); K_J=owned-worker-v833l.json
FF_FILES="ff_format.py ff_lifecycle.py ff_service.py affine_batch.py"
declare -A BASE=([ff_format.py]=a088d5ffb14897175cafd0ee76e0c9b74dae922932673a4e35c4b43f0b18528a [ff_lifecycle.py]=7b699a59fd59e7d03a44df7643e5d0d2a0767066edb0a35728c3a420b4d176c0
                 [ff_service.py]=179243fe14dd2ca7fe6d3741c278ffcf4653747d3a4d361f05daec907cb5cc10 [affine_batch.py]=a548d5cebff688dfeda49136d292150e2c01a695fe7c8646d2a861f0ab527aba)
NEED_GB=192; POD=/workspace/affine-mimo-trainer-prod; OW="$POD/runtime-exact/venv/bin/python -I -B $POD/owned-worker-v7.py"
TKH=/home/const/.lium/known_hosts/ef2b6085-adae-43e2-8a13-39182a3b3f7a
SSHO=(-T -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10 -o ServerAliveInterval=20 -o ServerAliveCountMax=6)
N(){ local i rc; for i in 1 2 3 4 5 6; do ssh "${SSHO[@]}" -o UserKnownHostsFile=$TKH -p 40022 root@8.28.35.244 "$@" && return 0; rc=$?; [[ $rc == 255 ]] || return $rc; sleep 5; done; return 255; }
cr(){ TZ=America/Costa_Rica date "+%H:%M:%S CR"; }
LOG=$AV/activate.log; log(){ echo "$(date -u +%FT%TZ) $(cr) [$MODE] $*" | tee -a "$LOG"; }
prog(){ echo "- $(TZ=America/Costa_Rica date '+%H:%M CR'): activate_sm $MODE: $*" >> "$D/PROGRESS.md"; }
die(){ log "ABORT: $*"; prog "ABORT: $*"; exit 1; }
sh256(){ sha256sum "$1" | cut -d' ' -f1; }
put(){ install -m "${3:-644}" "$1" "$2.sm-new"; mv -f "$2.sm-new" "$2"; }   # atomic replace
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
lastx(){ local X= c0 x; c0=$(cur); for x in $(seq $((c0-12)) $((c0+3))); do [[ -e $E/$x/TRAINING-RESULT.json ]] && X=$x; done; echo "$X"; }
kill_helpers(){ # PID-scoped: v83_dp helper PIDs (cmdline verified), TERM then KILL those exact PIDs only
  N 'pids=""; for p in $(pgrep -f "[h]ost-v833[jl]/v83_dp.py"); do tr "\0" " " </proc/$p/cmdline | grep -q "host-v833[jl]/v83_dp.py" && pids="$pids $p"; done
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
    local line new; line=$(grep '^ExecStart="/usr/bin/ssh"' "$TD/$J_TC"); new=${line//$J_J --sha256 $J_S/$K_J --sha256 $K_S}
    [[ "$new" != "$line" && "$new" == *"$K_J --sha256 $K_S"* ]] || return 15
    printf '%s\n%s\n%s\n%s\n%s\n' "# MiMo trainer v8.3.3l (= v8.3.3j + per-turn free-file train.step_mult + optional knobs, defaults = v8.3.3j). Jacob 2026-11-02 10:19 / 11:19 CR (duongna #120)." \
      "# Installed by $AV/activate_sm.sh $(cr). Revert: MODE=revert bash $AV/activate_sm.sh" "[Service]" "ExecStart=" "$new" > "$AV/$K_TC"
    install -m 600 "$AV/$K_TC" "$TD/$K_TC"
  else rm -f "$TD/$K_TC"; fi
  systemctl --user daemon-reload
  grep -q "$tj --sha256 $ts" <(systemctl --user show $SUP -p ExecStart --value) || return 16
  systemctl --user start $SUP
  ok=0; for i in $(seq 1 60); do N "$OW --config $POD/$tj --sha256 $ts --status" | grep -q '"alive": true' && { ok=1; break; }; sleep 5; done
  [[ $ok == 1 ]] || return 17
  log "trainer -> $tj ($ts) alive"; return 0; }
diskok(){ local f; f=$(N "df -B1G --output=avail $POD | tail -1" | tr -d ' '); [[ $f =~ ^[0-9]+$ ]] && (( f >= NEED_GB )) && { log "pod free ${f} GB >= $NEED_GB"; return 0; }; log "pod free ${f:-?} GB < $NEED_GB"; return 1; }
stage(){
  (cd $REL && sha256sum --quiet -c SIGNED-MANIFEST.sha256) || die "release manifest"
  [[ $(ls "$TD" | tail -1) == "$J_TC" ]] || die "v833j drop-in is not the newest trainer drop-in"
  N "df -T $POD | tail -1" | grep -qi fuse && die "pod path on fuse (gocryptfs) - refusing"
  tar cf - -C $REL trainer-host-v833l | N "set -e; cd $POD; if [ ! -d host-v833l ]; then rm -rf .host-v833l.new; mkdir -m 700 .host-v833l.new; tar --no-same-owner -xf - -C .host-v833l.new; chown -R root:root .host-v833l.new; mv .host-v833l.new/trainer-host-v833l host-v833l; rmdir .host-v833l.new; else cat >/dev/null; fi
    sed 's#/host-v833j/#/host-v833l/#g' $J_J > .$K_J.new; chmod 600 .$K_J.new; mv .$K_J.new $K_J"
  (cd $REL/trainer-host-v833l && find . -type f -print0 | sort -z | xargs -0 sha256sum) > $AV/host-v833l.sha256
  N "cd $POD/host-v833l && sha256sum --quiet -c -" < $AV/host-v833l.sha256 || die "pod host-v833l sha mismatch"
  N "cd $POD && diff <(sed 's#/host-v833l/#/host-v833j/#g' $K_J) $J_J >/dev/null && ! grep -q host-v833j $K_J" || die "worker config diff"
  N "cd $POD/source-final-v823u && env CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 MKL_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH=$POD/source-final-v823u TOKENIZERS_PARALLELISM=false timeout 240 $POD/runtime-exact/venv/bin/python -B $POD/host-v833l/mimo_host_trainer_v83.py --operator-config $POD/host-v833l/TRAINER-HOST.v8.3.3.ROOT-ADMITTED.private.json --check -- > /tmp/v833l-check.out 2>&1; rc=\$?; tail -c 1500 /tmp/v833l-check.out; exit \$rc" > $AV/check.out 2>&1 || die "v833l --check failed: $(tail -c 400 $AV/check.out)"
  K_S=$(N "sha256sum < $POD/$K_J" | cut -d' ' -f1); [[ $K_S =~ ^[0-9a-f]{64}$ ]] || die "K_S"; echo "$K_S" > $AV/K_S
  log "staged host-v833l + $K_J ($K_S)"
  local f; for f in $FF_FILES; do [[ $(sh256 $FFD/$f) == ${BASE[$f]} ]] || die "live $f changed since the build (sha $(sh256 $FFD/$f))"; done
  [[ $(sh256 $WEBFF/affine_batch.py) == ${BASE[affine_batch.py]} ]] || die "website affine_batch.py changed"
  rm -rf $AV/next; mkdir -m 700 $AV/next
  for f in $FF_FILES; do cp $STG/$f $AV/next/$f; done
  python3 $CODE/set_sm.py $AV/next/ff_format.py $AV/next/affine_batch.py >> $LOG || die "set_sm"
  sh256 $TPC > $AV/next/src-training_params.sha
  python3 $CODE/patch_tp_l.py $TPC $AV/next/training_params.py >> $LOG || die "training_params patch"
  python3 $CODE/make_ff_public_l.py $FFD/FF-PUBLIC.json $AV/next/FF-PUBLIC.json >> $LOG || die "FF-PUBLIC"
  (cd $AV/next && $PY -m py_compile training_params.py ff_format.py ff_lifecycle.py ff_service.py affine_batch.py) || die "compile"
  rm -rf $AV/next/__pycache__
  $PY - "$FFD/FF-PINS.json" "$AV/next" <<'EOF' || die "pins"
import json, hashlib, sys
p, nx = sys.argv[1], sys.argv[2]; d = json.load(open(p))
for n in ('ff_format.py', 'ff_lifecycle.py'): d['files'][n] = hashlib.sha256(open(nx + '/' + n, 'rb').read()).hexdigest()
open(nx + '/FF-PINS.json', 'w').write(json.dumps(d, indent=1) + '\n')
EOF
  log "host files built in $AV/next: $(cd $AV/next && sha256sum * | cut -c1-8 | tr '\n' ' ')"; }
case $MODE in
stage) stage; prog "staged (trainer v833l on pod, --check OK, worker config $(cat $AV/K_S); host files in activation/next)"; log done ;;
switch)
  [[ -f $TD/$K_TC ]] && die "v833l drop-in already installed"
  diskok || die "pod disk precheck failed"
  stage; K_S=$(cat $AV/K_S)
  ok=0; for i in $(seq 1 ${RWAIT:-4320}); do        # up to 12 h for a round boundary
    X=$(lastx); [[ -n $X && -f $GE/epochs/$X/DECISION.json ]] && grep -q '"decision": *"green"' $GE/epochs/$X/DECISION.json || { sleep 10; continue; }
    idle || { sleep 5; continue; }
    exec 8<>$STATE/lifecycle.lock; exec 9>$RA/LIFECYCLE-ACTIVATION.lock
    if flock -w 30 9 && flock -w 30 8; then
      DT=$(stat -c %Y $GE/epochs/$X/DECISION.json)
      other=$(find $RA -mindepth 3 -maxdepth 3 -path "$D" -prune -o \( -name ACTIVE.json -o -name LIVE.json \) -newermt "@$DT" -print 2>/dev/null | grep -v "^$D/" | head -1)
      if [[ -n $other ]]; then exec 8>&- 9>&-; log "round-boundary gap e$X already used by $other - waiting for the next GREEN"; sleep 60; continue; fi
      systemctl --user stop affine-mimo-lifecycle.service; c=$(cur)
      if ! openjob $c && [[ $(trainbusy) == 0 ]] && [[ $(lastx) == $X ]] && diskok; then ok=1; break; fi
      systemctl --user start affine-mimo-lifecycle.service; fi; exec 8>&- 9>&-
    sleep 5; done
  [[ $ok == 1 ]] || die "no idle round boundary within the wait"
  log "round boundary: e$X GREEN, trainer idle (epoch $(cur)), lifecycle stopped + locks held - switching"
  B=$AV/backup-$(date -u +%Y%m%dT%H%M%SZ); mkdir -m 700 $B; echo $B > $AV/LAST-BACKUP
  cp -a $TD $B/trainer-dropins; systemctl --user show $SUP -p ExecStart --value > $B/trainer-ExecStart.txt
  for f in $FF_FILES FF-PINS.json FF-PUBLIC.json; do cp -p $FFD/$f $B/$f; done
  cp -p $WEBFF/affine_batch.py $B/website-affine_batch.py; cp -p $TPC $B/training_params.py
  rc=0; switch_trainer $J_J $J_S $K_J $K_S install || rc=$?
  if [[ $rc != 0 ]]; then log "v833l switch failed rc=$rc - restoring v833j"; rm -f $TD/$K_TC; systemctl --user daemon-reload
    switch_trainer $K_J $K_S $J_J $J_S remove || log "WARNING restore rc=$?"; systemctl --user start affine-mimo-lifecycle.service; die "trainer switch failed rc=$rc (v833j restored)"; fi
  W=$(cur); while [[ -e $E/$W/TRAINING-JOB.json ]]; do W=$((W+1)); done      # first epoch whose job the v833l trainer runs
  if [[ $(sh256 $TPC) != $(cat $AV/next/src-training_params.sha) ]]; then python3 $CODE/patch_tp_l.py $TPC $AV/next/training_params.py >> $LOG || die "training_params re-patch"; cp -p $TPC $B/training_params.py; fi
  systemctl --user stop affine-mining-discovery.service                       # pins and files change together
  put $AV/next/FF-PINS.json $FFD/FF-PINS.json 600                             # re-pin BEFORE the pinned files change
  for f in $FF_FILES; do put $AV/next/$f $FFD/$f 644; done
  put $AV/next/affine_batch.py $WEBFF/affine_batch.py 644
  put $AV/next/FF-PUBLIC.json $FFD/FF-PUBLIC.json 644
  put $AV/next/training_params.py $TPC 644
  systemctl --user start affine-mimo-lifecycle.service; exec 8>&- 9>&-; log "lifecycle started (step_mult {$SM_ALLOWED} + knobs accepted), locks released"
  for u in affine-ff affine-mining-discovery affine-training-params affine-k1-transparency; do systemctl --user restart $u.service && log "$u restarted"; done
  echo "{\"from_epoch\": $W, \"trainer\": \"v833l\", \"sha256\": \"$K_S\", \"sm_allowed\": \"$SM_ALLOWED\", \"at\": \"$(cr)\", \"backup\": \"$B\", \"last_v833j_decided_epoch\": $X}" > $AV/ACTIVE.json
  systemctl --user reset-failed sm-watchdog.service 2>/dev/null || true
  systemd-run --user --unit sm-watchdog -p Restart=on-failure --setenv=FROM=$W --setenv=UNTIL=$(( $(date +%s) + 7*86400 )) $PY -u -B $AV/watchdog_sm.py >/dev/null
  log "sm-watchdog armed FROM=e$W (7 d)"
  python3 $CODE/web_docs_l.py apply $W >> $LOG 2>&1 && log "affine.io docs applied (from epoch $W)" || log "WARNING web docs apply failed (by hand: SM_ALLOWED=$SM_ALLOWED python3 $CODE/web_docs_l.py apply $W)"
  prog "LIVE v8.3.3l (step_mult {$SM_ALLOWED} + knobs) from epoch $W: trainer v833l ($K_S); last v833j turn decided e$X (GREEN, round boundary); backup $B"; log done ;;
revert)
  [[ -f $AV/ACTIVE.json ]] && mv -f $AV/ACTIVE.json $AV/REVERTED-$(date -u +%Y%m%dT%H%M%SZ).json
  B=$(cat $AV/LAST-BACKUP 2>/dev/null || true)
  if [[ ! -f $TD/$K_TC && ( -z $B || $(sh256 $FFD/ff_format.py) == ${BASE[ff_format.py]} ) ]]; then log "nothing installed - nothing to revert"; exit 0; fi
  K_S=$(cat $AV/K_S)
  if [[ ${FORCE:-0} != 1 ]]; then for i in $(seq 1 ${IWAIT:-720}); do idle && break; sleep 5; done; fi
  exec 9>$RA/LIFECYCLE-ACTIVATION.lock; flock -w 120 9 || true; exec 8<>$STATE/lifecycle.lock; flock -w 60 8 || true
  systemctl --user stop affine-mimo-lifecycle.service
  rc=0; if [[ -f $TD/$K_TC ]]; then switch_trainer $K_J $K_S $J_J $J_S remove || rc=$?; fi
  if [[ -n $B && -d $B ]]; then
    systemctl --user stop affine-mining-discovery.service
    put $B/FF-PINS.json $FFD/FF-PINS.json 600   # discovery requires 0600
    for f in $FF_FILES FF-PUBLIC.json; do put $B/$f $FFD/$f 644; done
    put $B/website-affine_batch.py $WEBFF/affine_batch.py 644
    [[ $(sh256 $TPC) == $(sh256 $AV/next/training_params.py 2>/dev/null || echo x) ]] && put $B/training_params.py $TPC 644 || log "training_params.py changed since the switch - left as is"
  fi
  systemctl --user start affine-mimo-lifecycle.service; exec 8>&- 9>&-
  for u in affine-ff affine-mining-discovery affine-training-params affine-k1-transparency; do systemctl --user restart $u.service || true; done
  if [[ -f $D/web/APPLIED.json ]]; then python3 $CODE/web_docs_l.py revert >> $LOG 2>&1 && log "affine.io docs restored" || log "WARNING web docs revert failed"; fi
  if [[ ${FROM_WATCHDOG:-0} != 1 ]]; then systemctl --user stop sm-watchdog.service 2>/dev/null || true; fi
  [[ $rc == 0 ]] || die "trainer revert failed rc=$rc (lifecycle restarted, ff files restored)"
  prog "REVERTED v8.3.3l (step_mult + knobs): trainer v833l -> v833j, ff files + pins + website client + FF-PUBLIC + training_params restored (${REASON:-manual})"; log done ;;
esac
