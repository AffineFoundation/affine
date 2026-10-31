#!/usr/bin/env bash
# RA/20261031-ff-train-params activation: optional per-turn lr / kl in free-file affine-batch-v2 files (Jacob 2026-10-31 07:08 CR, pt902u).
#   MODE=stage   verify + stage host-v833j and owned-worker-v833j.json on the trainer pod (+ --check); build the sidecar files. Inert.
#   MODE=switch  waits until v8.3.3i (RA/20261031-renorm-step-live) is live and stable for 2 turns, after 09:00 CR (ff-watchdog window),
#                for a gate DECISION gap no other activation used, and an idle trainer; then with the lifecycle stopped and both locks held:
#                trainer v833i -> v833j, FF-PINS re-pinned then ff_format / ff_lifecycle / ff_service / affine_batch installed,
#                FF-PUBLIC.json (mining.json batch_file), training_params.py and k1_transparency.py patched, FF-TRAIN.json enabled,
#                ff-watchdog (only v8.3.3i checks left) handed over to tp-watchdog.
#   MODE=revert  FF-TRAIN off at once, then (idle unless FORCE=1) everything back: trainer v833j -> v833i, ff files + pins, sidecars,
#                FF-PUBLIC; ff-watchdog re-armed with its original environment. One-line revert:
#                MODE=revert bash /home/const/subnet120-rewrite/state/root-audits/20261031-ff-train-params/activation/activate_tp.sh
set -euo pipefail; umask 077
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
MODE=${MODE:?MODE=stage|switch|revert}
RA=/home/const/subnet120-rewrite/state/root-audits; D=$RA/20261031-ff-train-params; AV=$D/activation; mkdir -p -m 700 $AV $D/state
REL=$D/release-v833j-signed; STATE=/home/const/subnet120-rewrite/state/mimo-mainnet-v1; E=$STATE/epochs
GE=$RA/20261026-promotion-gate/state; PY=/home/const/subnet120-rewrite/.venv/bin/python
RS=$RA/20261031-renorm-step-live; FFD=$RA/20261030-free-file/release; WEBFF=/home/const/subnet120/affine/website/ff
TPC=$RA/20261026-promotion-gate/code/training_params.py; K1T=$RA/20261027-k1-per-uid/code/k1_transparency.py
FLAG=$D/state/FF-TRAIN.json; SIDE=$D/side; STG=$D/stage/release
UNITS=/home/const/.config/systemd/user; SUP=affine-mimo-trainer-supervised.service; TD=$UNITS/$SUP.d
I_TC='zzzz~~~~~~~~~~~~~~~~~~v833i-owned-worker.conf'; J_TC='zzzz~~~~~~~~~~~~~~~~~~~v833j-owned-worker.conf'
I_J=owned-worker-v833i.json; I_S=$(cat $RS/activation/I_S); J_J=owned-worker-v833j.json
FF_FILES="ff_format.py ff_lifecycle.py ff_service.py affine_batch.py"
declare -A BASE=([ff_format.py]=b27dfac1cb4946e184f4b1625f905da24f4eddd43c6620aaaf7176d5142b9d62 [ff_lifecycle.py]=ee25f23f8b1d752d45a2bc4c7e49f7ebd97f78bd213cd4e1b8aa3c6e5ddca472
                 [ff_service.py]=451b2767ea0e05ab94caa53fce0ed76ff1437c11f7286ef3795559db6e7d35e1 [affine_batch.py]=47d7d1637045fd0d2b2631fb8740705a4c919409a21fc154ce293c7f4d00d390)
FF_WD_ENV='FROM=296 WATCH_UNTIL=1793458800 RS_UNTIL=1793538992'; RS_UNTIL=1793538992; FF_WATCH_UNTIL=1793458800
NEED_GB=192; POD=/workspace/affine-mimo-trainer-prod; OW="$POD/runtime-exact/venv/bin/python -I -B $POD/owned-worker-v7.py"
TKH=/home/const/.lium/known_hosts/ef2b6085-adae-43e2-8a13-39182a3b3f7a
SSHO=(-T -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10 -o ServerAliveInterval=20 -o ServerAliveCountMax=6)
N(){ local i rc; for i in 1 2 3 4 5 6; do ssh "${SSHO[@]}" -o UserKnownHostsFile=$TKH -p 40022 root@8.28.35.244 "$@" && return 0; rc=$?; [[ $rc == 255 ]] || return $rc; sleep 5; done; return 255; }
cr(){ TZ=America/Costa_Rica date "+%H:%M:%S CR"; }
LOG=$AV/activate.log; log(){ echo "$(date -u +%FT%TZ) $(cr) [$MODE] $*" | tee -a "$LOG"; }
prog(){ echo "- $(TZ=America/Costa_Rica date '+%H:%M CR'): activate_tp $MODE: $*" >> "$D/PROGRESS.md"; }
die(){ log "ABORT: $*"; prog "ABORT: $*"; exit 1; }
sh256(){ sha256sum "$1" | cut -d' ' -f1; }
put(){ install -m "${3:-644}" "$1" "$2.tp-new"; mv -f "$2.tp-new" "$2"; }   # atomic replace
setflag(){ printf '{"enabled": %s, "from_epoch": %s, "at": "%s", "by": "RA/20261031-ff-train-params activate_tp.sh %s"}\n' "$1" "$2" "$(cr)" "$MODE" > $FLAG.tmp; chmod 644 $FLAG.tmp; mv -f $FLAG.tmp $FLAG; }
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
  N 'pids=""; for p in $(pgrep -f "[h]ost-v833[ij]/v83_dp.py"); do tr "\0" " " </proc/$p/cmdline | grep -q "host-v833[ij]/v83_dp.py" && pids="$pids $p"; done
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
    local line new; line=$(grep '^ExecStart="/usr/bin/ssh"' "$TD/$I_TC"); new=${line//$I_J --sha256 $I_S/$J_J --sha256 $J_S}
    [[ "$new" != "$line" && "$new" == *"$J_J --sha256 $J_S"* ]] || return 15
    printf '%s\n%s\n%s\n%s\n%s\n' "# MiMo trainer v8.3.3j (= v8.3.3i + per-turn free-file lr / kl). Jacob 2026-10-31 07:08 CR (pt902u)." \
      "# Installed by $AV/activate_tp.sh $(cr). Revert: MODE=revert bash $AV/activate_tp.sh" "[Service]" "ExecStart=" "$new" > "$AV/$J_TC"
    install -m 600 "$AV/$J_TC" "$TD/$J_TC"
  else rm -f "$TD/$J_TC"; fi
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
  tar cf - -C $REL trainer-host-v833j | N "set -e; cd $POD; if [ ! -d host-v833j ]; then rm -rf .host-v833j.new; mkdir -m 700 .host-v833j.new; tar --no-same-owner -xf - -C .host-v833j.new; chown -R root:root .host-v833j.new; mv .host-v833j.new/trainer-host-v833j host-v833j; rmdir .host-v833j.new; else cat >/dev/null; fi
    sed 's#/host-v833i/#/host-v833j/#g' $I_J > .$J_J.new; chmod 600 .$J_J.new; mv .$J_J.new $J_J"
  (cd $REL/trainer-host-v833j && find . -type f -print0 | sort -z | xargs -0 sha256sum) > $AV/host-v833j.sha256
  N "cd $POD/host-v833j && sha256sum --quiet -c -" < $AV/host-v833j.sha256 || die "pod host-v833j sha mismatch"
  N "cd $POD && diff <(sed 's#/host-v833j/#/host-v833i/#g' $J_J) $I_J >/dev/null && ! grep -q host-v833i $J_J" || die "worker config diff"
  N "cd $POD/source-final-v823u && env CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 MKL_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH=$POD/source-final-v823u TOKENIZERS_PARALLELISM=false timeout 240 $POD/runtime-exact/venv/bin/python -B $POD/host-v833j/mimo_host_trainer_v83.py --operator-config $POD/host-v833j/TRAINER-HOST.v8.3.3.ROOT-ADMITTED.private.json --check -- > /tmp/v833j-check.out 2>&1; rc=\$?; tail -c 1500 /tmp/v833j-check.out; exit \$rc" > $AV/check.out 2>&1 || die "v833j --check failed: $(tail -c 400 $AV/check.out)"
  J_S=$(N "sha256sum < $POD/$J_J" | cut -d' ' -f1); [[ $J_S =~ ^[0-9a-f]{64}$ ]] || die "J_S"; echo "$J_S" > $AV/J_S
  log "staged host-v833j + $J_J ($J_S)"
  # host side: live ff files must still be the bases the change was built and tested on
  local f; for f in $FF_FILES; do [[ $(sh256 $FFD/$f) == ${BASE[$f]} ]] || die "live $f changed since the build (sha $(sh256 $FFD/$f))"; done
  [[ $(sh256 $WEBFF/affine_batch.py) == ${BASE[affine_batch.py]} ]] || die "website affine_batch.py changed"
  [[ $(sh256 $FFD/ff_launcher.py) == $(sh256 $STG/ff_launcher.py) ]] || die "ff_launcher.py changed"
  rm -rf $AV/next; mkdir -m 700 $AV/next
  for f in $FF_FILES; do cp $STG/$f $AV/next/$f; done
  sh256 $TPC > $AV/next/src-training_params.sha; sh256 $K1T > $AV/next/src-k1_transparency.sha
  $PY -B $SIDE/patch_sidecars.py training_params $TPC $AV/next/training_params.py >> $LOG || die "training_params patch"
  $PY -B $SIDE/patch_sidecars.py k1_transparency $K1T $AV/next/k1_transparency.py >> $LOG || die "k1_transparency patch"
  python3 $SIDE/make_ff_public.py $FFD/FF-PUBLIC.json $AV/next/FF-PUBLIC.json || die "FF-PUBLIC"
  (cd $AV/next && $PY -m py_compile training_params.py k1_transparency.py ff_format.py ff_lifecycle.py ff_service.py affine_batch.py) || die "compile"
  rm -rf $AV/next/__pycache__
  $PY - "$FFD/FF-PINS.json" "$AV/next" <<'EOF' || die "pins"
import json, hashlib, sys
p, nx = sys.argv[1], sys.argv[2]; d = json.load(open(p))
for n in ('ff_format.py', 'ff_lifecycle.py'): d['files'][n] = hashlib.sha256(open(nx + '/' + n, 'rb').read()).hexdigest()
open(nx + '/FF-PINS.json', 'w').write(json.dumps(d, indent=1) + '\n')
EOF
  log "host files built in $AV/next: $(cd $AV/next && sha256sum * | cut -c1-8 | tr '\n' ' ')"; }
stable2(){ # v8.3.3i live and 2 decided turns on it with clean stats; the renorm job not reverted
  $PY - "$RS/activation/ACTIVE.json" "$E" "$GE/epochs" <<'EOF'
import json, sys, math
from pathlib import Path
try: a = json.load(open(sys.argv[1]))
except Exception: print('v833i not live'); sys.exit(1)
if a.get('trainer') != 'v833i': print('v833i not live'); sys.exit(1)
E, G = Path(sys.argv[2]), Path(sys.argv[3]); ok = []
for e in range(int(a['from_epoch']), int(a['from_epoch']) + 40):
    r = E / str(e) / 'TRAINING-RESULT.json'
    if (E / str(e) / 'RECOVERY-REQUIRED.json').exists(): print('recovery e%d' % e); sys.exit(1)
    if not r.exists(): continue
    if not (G / str(e) / 'DECISION.json').exists(): continue
    j = json.loads(r.read_text()); j = j.get('payload', j); t = (j.get('report') or {}).get('training') or {}
    s = (t.get('optimizer') or {}).get('sgd') or {}; um = (s.get('fp32_master') or {}).get('update_norm_master')
    if t.get('status') == 'updated' and s.get('sgd_direction') == 'normgrad' and type(um) is float and math.isfinite(um): ok.append(e)
print('v833i stable turns: %s' % ok); sys.exit(0 if len(ok) >= 2 else 1)
EOF
  [[ $? == 0 ]] || return 1
  grep -q '"verdict": *"mismatch"' $RS/tests/EIGHT-VS-ONE.json 2>/dev/null && { echo '8v1 mismatch'; return 1; }
  [[ -f $RS/activation/ACTIVE.json ]] || { echo 'renorm ACTIVE.json gone'; return 1; }
  grep -Eiq "${RS_DONE_RE:-watch done|stays live|rollout complete|stable for 2 turns|2 stable turns|stable 2 turns|COMPLETE:|DONE:|final report}" $RS/PROGRESS.md || { echo 'renorm folder does not show the rollout complete yet'; return 1; }
  echo 'renorm rollout complete marker found'; }
case $MODE in
stage) stage; prog "staged (trainer v833j on pod, --check OK, worker config $(cat $AV/J_S); host files in activation/next)"; log done ;;
switch)
  [[ -f $TD/$J_TC ]] && die "v833j drop-in already installed"
  log "waiting: v8.3.3i live + 2 stable turns, and after 09:00 CR (ff-watchdog window)"
  for i in $(seq 1 ${SWAIT:-2160}); do
    if (( $(date +%s) >= FF_WATCH_UNTIL )) && stable2 > $AV/stable.out 2>&1; then break; fi; sleep 10; done
  stable2 > $AV/stable.out 2>&1 || die "v8.3.3i not stable: $(cat $AV/stable.out)"
  log "$(cat $AV/stable.out)"
  grep -q "$I_J --sha256 $I_S" <(systemctl --user show $SUP -p ExecStart --value) || die "live trainer is not v833i"
  [[ $(ls $TD | tail -1) == "$I_TC" ]] || die "unexpected newer trainer drop-in"
  [[ -f $REL/SIGNING-RESULT.json ]] || die "release-v833j-signed missing (run sign_v833j.py)"
  diskok || die "pod disk precheck failed"
  stage; J_S=$(cat $AV/J_S)
  X=; c0=$(cur); for x in $(seq $((c0-12)) $((c0+3))); do [[ -e $E/$x/TRAINING-RESULT.json ]] && X=$x; done   # last trained epoch (odd epochs are often ABANDONED-SPECULATIVE)
  [[ -n $X ]] || die "no TRAINING-RESULT in e$((c0-12))..e$((c0+3))"
  if [[ ! -f $GE/epochs/$X/DECISION.json ]]; then log "waiting for gate DECISION e$X (+ RED restore) before switching"
    while [[ ! -f $GE/epochs/$X/DECISION.json ]]; do sleep 1; done
    if grep -q '"decision": *"red"' $GE/epochs/$X/DECISION.json; then
      for i in $(seq 1 450); do [[ -f $GE/epochs/$X/TRAINER-RESTORED.json ]] && break; sleep 2; done; sleep 3; fi
  fi
  ok=0; for i in $(seq 1 ${IWAIT:-720}); do
    if idle; then exec 8<>$STATE/lifecycle.lock; exec 9>$RA/LIFECYCLE-ACTIVATION.lock
      if flock -w 30 9 && flock -w 30 8; then
        DT=$(stat -c %Y $GE/epochs/$X/DECISION.json)
        other=$(find $RS $RA/20261031-legacy-removal $RA/20261030-heldout-gate $RA/20261031-gate-noise-floor -maxdepth 2 \( -name 'activate*.log' -o -name 'step*.log' -o -name 'apply*.log' -o -name ACTIVE.json -o -name LIVE.json \) -newermt "@$DT" 2>/dev/null | head -1)
        if [[ -n $other ]]; then exec 8>&- 9>&-; log "decision gap e$X already used by $other - retrying at the next gap"; prog "gap e$X taken by $other, waiting"
          sleep 300; exec env MODE=switch bash "$0"; fi
        systemctl --user stop affine-mimo-lifecycle.service; c=$(cur)
        if ! openjob $c && [[ $(trainbusy) == 0 ]] && diskok; then ok=1; break; fi
        systemctl --user start affine-mimo-lifecycle.service; fi; exec 8>&- 9>&-; fi
    sleep 5; done
  [[ $ok == 1 ]] || die "trainer never idle"
  log "trainer idle (epoch $(cur), last decided e$X), lifecycle stopped + locks held - switching"
  B=$AV/backup-$(date -u +%Y%m%dT%H%M%SZ); mkdir -m 700 $B; echo $B > $AV/LAST-BACKUP
  cp -a $TD $B/trainer-dropins; systemctl --user show $SUP -p ExecStart --value > $B/trainer-ExecStart.txt
  for f in $FF_FILES FF-PINS.json FF-PUBLIC.json; do cp -p $FFD/$f $B/$f; done
  cp -p $WEBFF/affine_batch.py $B/website-affine_batch.py; cp -p $TPC $B/training_params.py; cp -p $K1T $B/k1_transparency.py
  systemctl --user show ff-watchdog -p Environment --value > $B/ff-watchdog.env 2>/dev/null || true
  rc=0; switch_trainer $I_J $I_S $J_J $J_S install || rc=$?
  if [[ $rc != 0 ]]; then log "v833j switch failed rc=$rc - restoring v833i"; rm -f $TD/$J_TC; systemctl --user daemon-reload
    switch_trainer $J_J $J_S $I_J $I_S remove || log "WARNING restore rc=$?"; systemctl --user start affine-mimo-lifecycle.service; die "trainer switch failed rc=$rc (v833i restored)"; fi
  W=$(cur); while [[ -e $E/$W/TRAINING-JOB.json ]]; do W=$((W+1)); done      # first epoch whose job the v833j trainer runs
  if [[ $(sh256 $TPC) != $(cat $AV/next/src-training_params.sha) ]]; then $PY -B $SIDE/patch_sidecars.py training_params $TPC $AV/next/training_params.py >> $LOG || die "training_params re-patch"; log "training_params.py changed since stage: re-patched the live copy"; fi
  if [[ $(sh256 $K1T) != $(cat $AV/next/src-k1_transparency.sha) ]]; then $PY -B $SIDE/patch_sidecars.py k1_transparency $K1T $AV/next/k1_transparency.py >> $LOG || die "k1_transparency re-patch"; log "k1_transparency.py changed since stage: re-patched the live copy"; fi
  cp -p $TPC $B/training_params.py; cp -p $K1T $B/k1_transparency.py
  setflag true $W
  systemctl --user stop affine-mining-discovery.service                       # pins and files change together
  put $AV/next/FF-PINS.json $FFD/FF-PINS.json 600                             # re-pin BEFORE the pinned files change
  for f in ff_format.py ff_lifecycle.py ff_service.py affine_batch.py; do put $AV/next/$f $FFD/$f 644; done
  put $AV/next/affine_batch.py $WEBFF/affine_batch.py 644
  put $AV/next/FF-PUBLIC.json $FFD/FF-PUBLIC.json 644
  put $AV/next/training_params.py $TPC 644; put $AV/next/k1_transparency.py $K1T 644
  systemctl --user start affine-mimo-lifecycle.service; exec 8>&- 9>&-; log "lifecycle started (ff_lifecycle v2), locks released"
  for u in affine-ff affine-mining-discovery affine-training-params affine-k1-transparency; do systemctl --user restart $u.service && log "$u restarted"; done
  echo "{\"from_epoch\": $W, \"trainer\": \"v833j\", \"sha256\": \"$J_S\", \"at\": \"$(cr)\", \"backup\": \"$B\", \"last_v833i_decided_epoch\": $X}" > $AV/ACTIVE.json
  if systemctl --user is-active --quiet ff-watchdog.service; then systemctl --user stop ff-watchdog.service; log "ff-watchdog stopped (only v8.3.3i checks were left; tp-watchdog covers them)"; fi
  systemctl --user reset-failed tp-watchdog.service 2>/dev/null || true
  systemd-run --user --unit tp-watchdog -p Restart=on-failure --setenv=FROM=$W --setenv=UNTIL=$(( $(date +%s) + 86400 )) $PY -u -B $AV/watchdog_tp.py >/dev/null
  log "tp-watchdog armed FROM=e$W (24 h)"
  prog "LIVE lr/kl from epoch $W: trainer v833j ($J_S), affine-batch-v2 accepted (FF-TRAIN on); last v833i turn decided e$X; backup $B"; log done ;;
revert)
  [[ -f $FLAG ]] && { setflag false null; log "FF-TRAIN off (v2 files invalid, no train in new dispatches)"; }
  [[ -f $AV/ACTIVE.json ]] && mv -f $AV/ACTIVE.json $AV/REVERTED-$(date -u +%Y%m%dT%H%M%SZ).json
  B=$(cat $AV/LAST-BACKUP 2>/dev/null || true)
  if [[ ! -f $TD/$J_TC && ( -z $B || $(sh256 $FFD/ff_lifecycle.py) == ${BASE[ff_lifecycle.py]} ) ]]; then log "nothing installed - nothing to revert"; exit 0; fi
  J_S=$(cat $AV/J_S)
  if [[ ${FORCE:-0} != 1 ]]; then for i in $(seq 1 ${IWAIT:-720}); do idle && break; sleep 5; done; fi
  exec 9>$RA/LIFECYCLE-ACTIVATION.lock; flock -w 120 9 || true; exec 8<>$STATE/lifecycle.lock; flock -w 60 8 || true
  systemctl --user stop affine-mimo-lifecycle.service
  rc=0; if [[ -f $TD/$J_TC ]]; then switch_trainer $J_J $J_S $I_J $I_S remove || rc=$?; fi
  if [[ -n $B && -d $B ]]; then
    systemctl --user stop affine-mining-discovery.service
    put $B/FF-PINS.json $FFD/FF-PINS.json 600   # discovery requires 0600
    for f in $FF_FILES FF-PUBLIC.json; do put $B/$f $FFD/$f 644; done
    put $B/website-affine_batch.py $WEBFF/affine_batch.py 644
    [[ $(sh256 $TPC) == $(sh256 $AV/next/training_params.py 2>/dev/null || echo x) ]] && put $B/training_params.py $TPC 644 || log "training_params.py changed since the switch - left as is (FF-TRAIN off hides the block)"
    [[ $(sh256 $K1T) == $(sh256 $AV/next/k1_transparency.py 2>/dev/null || echo x) ]] && put $B/k1_transparency.py $K1T 644 || log "k1_transparency.py changed since the switch - left as is"
  fi
  systemctl --user start affine-mimo-lifecycle.service; exec 8>&- 9>&-
  for u in affine-ff affine-mining-discovery affine-training-params affine-k1-transparency; do systemctl --user restart $u.service || true; done
  if (( $(date +%s) < RS_UNTIL )) && ! systemctl --user is-active --quiet ff-watchdog.service && grep -q '"v833i"' $RS/activation/ACTIVE.json 2>/dev/null; then
    systemctl --user reset-failed ff-watchdog.service 2>/dev/null || true
    systemd-run --user --unit ff-watchdog -p Restart=on-failure --setenv=FROM=296 --setenv=WATCH_UNTIL=$FF_WATCH_UNTIL --setenv=RS_UNTIL=$RS_UNTIL \
      $PY -u -B $RA/20261030-free-file/activation/watchdog_ff.py >/dev/null && log "ff-watchdog re-armed ($FF_WD_ENV)"; fi
  [[ $rc == 0 ]] || die "trainer revert failed rc=$rc (lifecycle restarted, ff files restored)"
  prog "REVERTED lr/kl: trainer v833j -> v833i, ff files + pins + sidecars restored (${REASON:-manual})"; log done ;;
esac
