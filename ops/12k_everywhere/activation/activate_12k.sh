#!/usr/bin/env bash
# RA/20261102-12k-everywhere activation: 12,288 output tokens everywhere (Jacob 2026-11-02 18:48 CR "12 k should be everywhere").
#   trainer v8.3.3l -> v8.3.3m (= v8.3.3l + host context limit max(runtime, 16384) + free-file overlong window 10096..12288, same 0.5 / 2192 ramp);
#   ff_format MAX_OUTPUT 12288 / MAX_CONTEXT 16384 (+ LEGACY_CAP 8192: queued 8K-generated files stay valid, exactly 8192 = capped);
#   website affine_batch.py, FF-PUBLIC.json, FF-PINS re-pin, training_params.py (mining.json), affine.io llms.txt.
#   MODE=stage   verify + stage host-v833m + owned-worker-v833m.json on the trainer pod (--check); build host files in activation/next. Inert.
#   MODE=switch  in an idle decision gap (no open training job, gate decided, no restore running), lifecycle stopped and both locks held:
#                switch trainer, install files, restart units, arm 12k-watchdog (auto-revert on trainer OOM / epoch failure).
#   MODE=revert  everything back (idle unless FORCE=1). One-line revert:
#                MODE=revert bash /home/const/subnet120-rewrite/state/root-audits/20261102-12k-everywhere/activation/activate_12k.sh
set -euo pipefail; umask 077
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
MODE=${MODE:?MODE=stage|switch|revert}
RA=/home/const/subnet120-rewrite/state/root-audits; D=$RA/20261102-12k-everywhere; AV=$D/activation; mkdir -p -m 700 $AV
REL=$D/release-v833m-signed; STG=$D/stage/ff; STATE=/home/const/subnet120-rewrite/state/mimo-mainnet-v1; E=$STATE/epochs
GE=$RA/20261026-promotion-gate/state; PY=/home/const/subnet120-rewrite/.venv/bin/python
FFD=$RA/20261030-free-file/release; WEBFF=/home/const/subnet120/affine/website/ff
TPC=$RA/20261026-promotion-gate/code/training_params.py
UNITS=/home/const/.config/systemd/user; SUP=affine-mimo-trainer-supervised.service; TD=$UNITS/$SUP.d
L_TC='zzzz~~~~~~~~~~~~~~~~~~~~v833l-owned-worker.conf'; M_TC='zzzz~~~~~~~~~~~~~~~~~~~~~v833m-owned-worker.conf'
L_J=owned-worker-v833l.json; L_S=$(cat $RA/20261102-step-mult/activation/K_S); M_J=owned-worker-v833m.json
FF_FILES="ff_format.py affine_batch.py"
declare -A BASE=([ff_format.py]=4430cec7f0c37012d1257c80bb8cd590428e0b6054afc73e8e2cc5cb01298249 [affine_batch.py]=ed6aa8a18c69e385aa185f76d928735595bde1ea9627dea293570b21733fda79
                 [FF-PUBLIC.json]=7277b78c0459adcc489b241b5f25b0e0d614b05e72a1c58c8094c809cddd3ef9)
NEED_GB=192; POD=/workspace/affine-mimo-trainer-prod; OW="$POD/runtime-exact/venv/bin/python -I -B $POD/owned-worker-v7.py"
TSH=/home/const/subnet120-rewrite/state/root-audits/20261026-rollback-189de412/scripts/rssh.sh   # repointed by replace_pod.py
N(){ local i rc; for i in 1 2 3 4 5 6; do bash $TSH "$@" && return 0; rc=$?; [[ $rc == 255 ]] || return $rc; sleep 5; done; return 255; }
cr(){ TZ=America/Costa_Rica date "+%H:%M:%S CR"; }
LOG=$AV/activate.log; log(){ echo "$(date -u +%FT%TZ) $(cr) [$MODE] $*" | tee -a "$LOG"; }
prog(){ echo "- $(TZ=America/Costa_Rica date '+%Y-%m-%d %H:%M CR'): activate_12k $MODE: $*" >> "$D/PROGRESS.md"; }
die(){ log "ABORT: $*"; prog "ABORT: $*"; exit 1; }
sh256(){ sha256sum "$1" | cut -d' ' -f1; }
put(){ install -m "${3:-644}" "$1" "$2.12k-new"; mv -f "$2.12k-new" "$2"; }
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
kill_helpers(){
  N 'pids=""; for p in $(pgrep -f "[h]ost-v833[lm]/v83_dp.py"); do tr "\0" " " </proc/$p/cmdline | grep -q "host-v833[lm]/v83_dp.py" && pids="$pids $p"; done
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
    local line new; line=$(grep '^ExecStart="/usr/bin/ssh"' "$TD/$L_TC"); new=${line//$L_J --sha256 $L_S/$M_J --sha256 $M_S}
    [[ "$new" != "$line" && "$new" == *"$M_J --sha256 $M_S"* ]] || return 15
    printf '%s\n%s\n%s\n%s\n%s\n' "# MiMo trainer v8.3.3m (= v8.3.3l + 12K: host context max(runtime,16384), ff overlong window 10096..12288). Jacob 2026-11-02 18:48 CR." \
      "# Installed by $AV/activate_12k.sh $(cr). Revert: MODE=revert bash $AV/activate_12k.sh" "[Service]" "ExecStart=" "$new" > "$AV/$M_TC"
    install -m 600 "$AV/$M_TC" "$TD/$M_TC"
  else rm -f "$TD/$M_TC"; fi
  systemctl --user daemon-reload
  grep -q "$tj --sha256 $ts" <(systemctl --user show $SUP -p ExecStart --value) || return 16
  systemctl --user start $SUP
  ok=0; for i in $(seq 1 60); do N "$OW --config $POD/$tj --sha256 $ts --status" | grep -q '"alive": true' && { ok=1; break; }; sleep 5; done
  [[ $ok == 1 ]] || return 17
  log "trainer -> $tj ($ts) alive"; return 0; }
diskok(){ local f; f=$(N "df -B1G --output=avail $POD | tail -1" | tr -d ' '); [[ $f =~ ^[0-9]+$ ]] && (( f >= NEED_GB )) && { log "pod free ${f} GB >= $NEED_GB"; return 0; }; log "pod free ${f:-?} GB < $NEED_GB"; return 1; }
stage(){
  (cd $REL && sha256sum --quiet -c SIGNED-MANIFEST.sha256) || die "release manifest"
  [[ $(ls "$TD" | tail -1) == "$L_TC" ]] || die "v833l drop-in is not the newest trainer drop-in"
  N "df -T $POD | tail -1" | grep -qi fuse && die "pod path on fuse - refusing"
  tar cf - -C $REL trainer-host-v833m | N "set -e; cd $POD; if [ ! -d host-v833m ]; then rm -rf .host-v833m.new; mkdir -m 700 .host-v833m.new; tar --no-same-owner -xf - -C .host-v833m.new; chown -R root:root .host-v833m.new; mv .host-v833m.new/trainer-host-v833m host-v833m; rmdir .host-v833m.new; else cat >/dev/null; fi
    sed 's#/host-v833l/#/host-v833m/#g' $L_J > .$M_J.new; chmod 600 .$M_J.new; mv .$M_J.new $M_J"
  (cd $REL/trainer-host-v833m && find . -type f -print0 | sort -z | xargs -0 sha256sum) > $AV/host-v833m.sha256
  N "cd $POD/host-v833m && sha256sum --quiet -c -" < $AV/host-v833m.sha256 || die "pod host-v833m sha mismatch"
  N "cd $POD && diff <(sed 's#/host-v833m/#/host-v833l/#g' $M_J) $L_J >/dev/null && ! grep -q host-v833l $M_J" || die "worker config diff"
  N "cd $POD/source-final-v823u && env CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 MKL_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH=$POD/source-final-v823u TOKENIZERS_PARALLELISM=false timeout 240 $POD/runtime-exact/venv/bin/python -B $POD/host-v833m/mimo_host_trainer_v83.py --operator-config $POD/host-v833m/TRAINER-HOST.v8.3.3.ROOT-ADMITTED.private.json --check -- > /tmp/v833m-check.out 2>&1; rc=\$?; tail -c 1500 /tmp/v833m-check.out; exit \$rc" > $AV/check.out 2>&1 || die "v833m --check failed: $(tail -c 400 $AV/check.out)"
  M_S=$(N "sha256sum < $POD/$M_J" | cut -d' ' -f1); [[ $M_S =~ ^[0-9a-f]{64}$ ]] || die "M_S"; echo "$M_S" > $AV/M_S
  log "staged host-v833m + $M_J ($M_S)"
  local f; for f in $FF_FILES FF-PUBLIC.json; do [[ $(sh256 $FFD/$f) == ${BASE[$f]} ]] || die "live $f changed since the build (sha $(sh256 $FFD/$f))"; done
  [[ $(sh256 $WEBFF/affine_batch.py) == ${BASE[affine_batch.py]} ]] || die "website affine_batch.py changed"
  rm -rf $AV/next; mkdir -m 700 $AV/next
  python3 $D/release/patch_ff_12k.py $FFD $AV/next >> $LOG || die "ff patch"
  sh256 $TPC > $AV/next/src-training_params.sha
  python3 $D/release/patch_tp_12k.py $TPC $AV/next/training_params.py >> $LOG || die "training_params patch"
  (cd $AV/next && $PY -m py_compile training_params.py ff_format.py affine_batch.py) || die "compile"
  rm -rf $AV/next/__pycache__
  $PY - "$FFD/FF-PINS.json" "$AV/next" "$FFD" <<'EOF' || die "pins"
import json, hashlib, sys
p, nx, ffd = sys.argv[1], sys.argv[2], sys.argv[3]; d = json.load(open(p))
assert d['files']['ff_lifecycle.py'] == hashlib.sha256(open(ffd + '/ff_lifecycle.py', 'rb').read()).hexdigest()
d['files']['ff_format.py'] = hashlib.sha256(open(nx + '/ff_format.py', 'rb').read()).hexdigest()
open(nx + '/FF-PINS.json', 'w').write(json.dumps(d, indent=1) + '\n')
EOF
  python3 $D/activation/web_docs_12k.py build 0 $AV/next/web >> $LOG || die "web docs build"
  log "host files built in $AV/next: $(cd $AV/next && sha256sum * 2>/dev/null | cut -c1-8 | tr '\n' ' ')"; }
case $MODE in
stage) stage; prog "staged (trainer v833m on pod, --check OK, worker config $(cat $AV/M_S); host files in activation/next)"; log done ;;
switch)
  [[ -f $TD/$M_TC ]] && die "v833m drop-in already installed"
  diskok || die "pod disk precheck failed"
  stage; M_S=$(cat $AV/M_S)
  ok=0; for i in $(seq 1 ${RWAIT:-4320}); do        # up to 12 h for an idle decision gap
    X=$(lastx); [[ -n $X && -f $GE/epochs/$X/DECISION.json ]] || { sleep 10; continue; }
    idle || { sleep 5; continue; }
    exec 8<>$STATE/lifecycle.lock; exec 9>$RA/LIFECYCLE-ACTIVATION.lock
    if flock -w 30 9 && flock -w 30 8; then
      systemctl --user stop affine-mimo-lifecycle.service; c=$(cur)
      if ! openjob $c && [[ $(trainbusy) == 0 ]] && [[ $(lastx) == $X ]] && diskok; then ok=1; break; fi
      systemctl --user start affine-mimo-lifecycle.service; fi; exec 8>&- 9>&-
    sleep 5; done
  [[ $ok == 1 ]] || die "no idle decision gap within the wait"
  log "decision gap: e$X decided ($(grep -o '"decision": *"[a-z]*"' $GE/epochs/$X/DECISION.json)), trainer idle (epoch $(cur)), lifecycle stopped + locks held - switching"
  B=$AV/backup-$(date -u +%Y%m%dT%H%M%SZ); mkdir -m 700 $B; echo $B > $AV/LAST-BACKUP
  cp -a $TD $B/trainer-dropins; systemctl --user show $SUP -p ExecStart --value > $B/trainer-ExecStart.txt
  for f in $FF_FILES FF-PINS.json FF-PUBLIC.json; do cp -p $FFD/$f $B/$f; done
  cp -p $WEBFF/affine_batch.py $B/website-affine_batch.py; cp -p $TPC $B/training_params.py
  rc=0; switch_trainer $L_J $L_S $M_J $M_S install || rc=$?
  if [[ $rc != 0 ]]; then log "v833m switch failed rc=$rc - restoring v833l"; rm -f $TD/$M_TC; systemctl --user daemon-reload
    switch_trainer $M_J $M_S $L_J $L_S remove || log "WARNING restore rc=$?"; systemctl --user start affine-mimo-lifecycle.service; die "trainer switch failed rc=$rc (v833l restored)"; fi
  W=$(cur); while [[ -e $E/$W/TRAINING-JOB.json ]]; do W=$((W+1)); done      # first epoch whose job the v833m trainer runs
  if [[ $(sh256 $TPC) != $(cat $AV/next/src-training_params.sha) ]]; then python3 $D/release/patch_tp_12k.py $TPC $AV/next/training_params.py >> $LOG || die "training_params re-patch"; cp -p $TPC $B/training_params.py; fi
  systemctl --user stop affine-mining-discovery.service
  put $AV/next/FF-PINS.json $FFD/FF-PINS.json 600
  for f in $FF_FILES; do put $AV/next/$f $FFD/$f 644; done
  put $AV/next/affine_batch.py $WEBFF/affine_batch.py 644
  put $AV/next/FF-PUBLIC.json $FFD/FF-PUBLIC.json 644
  put $AV/next/training_params.py $TPC 644
  systemctl --user start affine-mimo-lifecycle.service; exec 8>&- 9>&-; log "lifecycle started (12K batch files accepted), locks released"
  for u in affine-ff affine-mining-discovery affine-training-params affine-k1-transparency; do systemctl --user restart $u.service && log "$u restarted"; done
  OFF=$(N "stat -c %s $POD/owned-worker-v7.log" 2>/dev/null || echo 0)
  echo "{\"from_epoch\": $W, \"trainer\": \"v833m\", \"sha256\": \"$M_S\", \"at\": \"$(cr)\", \"backup\": \"$B\", \"last_v833l_decided_epoch\": $X, \"log_offset\": $OFF}" > $AV/ACTIVE.json
  systemctl --user reset-failed 12k-watchdog.service 2>/dev/null || true
  systemd-run --user --unit 12k-watchdog -p Restart=on-failure --setenv=FROM=$W --setenv=OFF=$OFF --setenv=UNTIL=$(( $(date +%s) + 3*86400 )) $PY -u -B $AV/watchdog_12k.py >/dev/null
  log "12k-watchdog armed FROM=e$W (3 d)"
  python3 $D/activation/web_docs_12k.py apply $W >> $LOG 2>&1 && log "affine.io llms.txt applied (from epoch $W)" || log "WARNING web docs apply failed (by hand: python3 $D/activation/web_docs_12k.py apply $W)"
  prog "LIVE 12K everywhere from epoch $W: trainer v833m ($M_S), ff_format 12288/16384 (+8192 legacy capped), FF-PUBLIC, training_params, llms.txt; last v833l epoch decided e$X; backup $B"; log done ;;
revert)
  [[ -f $AV/ACTIVE.json ]] && mv -f $AV/ACTIVE.json $AV/REVERTED-$(date -u +%Y%m%dT%H%M%SZ).json
  B=$(cat $AV/LAST-BACKUP 2>/dev/null || true)
  if [[ ! -f $TD/$M_TC && ( -z $B || $(sh256 $FFD/ff_format.py) == ${BASE[ff_format.py]} ) ]]; then log "nothing installed - nothing to revert"; exit 0; fi
  M_S=$(cat $AV/M_S)
  if [[ ${FORCE:-0} != 1 ]]; then for i in $(seq 1 ${IWAIT:-720}); do idle && break; sleep 5; done; fi
  exec 9>$RA/LIFECYCLE-ACTIVATION.lock; flock -w 120 9 || true; exec 8<>$STATE/lifecycle.lock; flock -w 60 8 || true
  systemctl --user stop affine-mimo-lifecycle.service
  rc=0; if [[ -f $TD/$M_TC ]]; then switch_trainer $M_J $M_S $L_J $L_S remove || rc=$?; fi
  if [[ -n $B && -d $B ]]; then
    systemctl --user stop affine-mining-discovery.service
    put $B/FF-PINS.json $FFD/FF-PINS.json 600
    for f in $FF_FILES FF-PUBLIC.json; do put $B/$f $FFD/$f 644; done
    put $B/website-affine_batch.py $WEBFF/affine_batch.py 644
    [[ $(sh256 $TPC) == $(sh256 $AV/next/training_params.py 2>/dev/null || echo x) ]] && put $B/training_params.py $TPC 644 || log "training_params.py changed since the switch - left as is"
  fi
  systemctl --user start affine-mimo-lifecycle.service; exec 8>&- 9>&-
  for u in affine-ff affine-mining-discovery affine-training-params affine-k1-transparency; do systemctl --user restart $u.service || true; done
  if [[ -f $D/web/APPLIED.json ]]; then python3 $D/activation/web_docs_12k.py revert >> $LOG 2>&1 && log "affine.io llms.txt restored" || log "WARNING web docs revert failed"; fi
  if [[ ${FROM_WATCHDOG:-0} != 1 ]]; then systemctl --user stop 12k-watchdog.service 2>/dev/null || true; fi
  [[ $rc == 0 ]] || die "trainer revert failed rc=$rc (lifecycle restarted, ff files restored)"
  prog "REVERTED 12K: trainer v833m -> v833l, ff files + pins + website client + FF-PUBLIC + training_params + llms.txt restored (${REASON:-manual})"; log done ;;
esac
