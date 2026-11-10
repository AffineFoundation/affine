#!/usr/bin/env bash
# RA/20261030-free-file activation: SN120 free-file miner protocol (Jacob 2026-10-30 pt898u).
#   MODE=stage   verify + stage everything that is inert until the switch: trainer host-v833h on the pod (+ --check),
#                FF-PINS.json, staged FF.json (enabled false), affine-ff.service (links + readiness), website/ff/affine_batch.py.
#   MODE=switch  waits for the gate DECISION of the epoch in training/gating, then for an idle trainer; with the lifecycle stopped
#                and both locks held: trainer v833g -> v833h, patched sidecars, lifecycle -> release/ff_launcher.py, FF.json enabled
#                from the current epoch, PROMOTION-GATE turn_batch_counted_from_epoch; then starts ff-watchdog.
#   MODE=revert  restores every replaced file and unit (waits for an idle trainer unless FORCE=1). One-line revert:
#                MODE=revert bash /home/const/subnet120-rewrite/state/root-audits/20261030-free-file/activation/activate_ff.sh
set -euo pipefail; umask 077
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
MODE=${MODE:?MODE=stage|switch|restart|revert}
RA=/home/const/subnet120-rewrite/state/root-audits; D=$RA/20261030-free-file; AV=$D/activation; R=$D/release; mkdir -p -m 700 $AV $D/state
REL=$D/release-v833h-signed; STATE=/home/const/subnet120-rewrite/state/mimo-mainnet-v1; E=$STATE/epochs
STAGED=$RA/20261023-toploc-credit-train-gate/staged; FF=$STAGED/FF.json; PG=$STAGED/PROMOTION-GATE.json
GE=$RA/20261026-promotion-gate/state; WEB=/home/const/subnet120/affine/website
PY=/home/const/subnet120-rewrite/.venv/bin/python
UNITS=/home/const/.config/systemd/user; SUP=affine-mimo-trainer-supervised.service; TD=$UNITS/$SUP.d
LD=$UNITS/affine-mimo-lifecycle.service.d; L_TC='zzzz~~~~~~~~~~~~~~v829-free-file.conf'
G_TC='zzzz~~~~~~~~~~~~~~~~v833g-owned-worker.conf'; H_TC='zzzz~~~~~~~~~~~~~~~~~v833h-owned-worker.conf'
G_J=owned-worker-v833g.json; G_S=01329f0a3b4386fabb8d24230bb40832c9d69ebd456acdc1c8bf24525ce403d6; H_J=owned-worker-v833h.json
L=$RA/20261029-v829u-125task/release-v829x-v823u-signed
L_ARGS="--operator-config $L/v829-operator.v8.2.3u.ROOT-ADMITTED.private.json --v828-operator-config $L/v828-operator.v8.2.3u.ROOT-ADMITTED.private.json"
CUTOFF=1793437200; TOK=$RA/20261030-determinism/tok
# live sidecar files replaced by the switch: name|path|unit|sha256 of the version the patch was made against
SVC="gate_service|$RA/20261026-promotion-gate/code/gate_service.py|affine-promotion-gate|67679d75729cffc0369c41d61eb283b250bb4a2860a15377dff1be76f26bcd8a
training_params|$RA/20261026-promotion-gate/code/training_params.py|affine-training-params|40bbdbff84acee9d64c05f4ac8b5ed56c7bd73c45792adb323e2c0dcff19927f
k1_service|$RA/20261027-k1-per-uid/code/k1_service.py|affine-k1|2c8181ec82b5e4bed5d74a2965f634e5d0afb5dbe582993588a084e67527393c
k1_transparency|$RA/20261027-k1-per-uid/code/k1_transparency.py|affine-k1-transparency|cf3fb38c8f91763f4340e2e769fac5969311fda02055154a9f31453f5cc539b7
publish_mimo_public_discovery|$RA/20261023-toploc-credit-train-gate/observer-v829/publish_mimo_public_discovery.py|affine-mining-discovery|f7b02d84fa488154d037234ab612f2a27509ce75cdfcd026d2739831c3032615"
NEED_GB=192; POD=/workspace/affine-mimo-trainer-prod; OW="$POD/runtime-exact/venv/bin/python -I -B $POD/owned-worker-v7.py"
TKH=/home/const/.lium/known_hosts/ef2b6085-adae-43e2-8a13-39182a3b3f7a
SSHO=(-T -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10 -o ServerAliveInterval=20 -o ServerAliveCountMax=6)
N(){ local i rc; for i in 1 2 3 4 5 6; do ssh "${SSHO[@]}" -o UserKnownHostsFile=$TKH -p 40022 root@8.28.35.244 "$@" && return 0; rc=$?; [[ $rc == 255 ]] || return $rc; sleep 5; done; return 255; }
cr(){ TZ=America/Costa_Rica date "+%H:%M:%S CR"; }
LOG=$AV/activate.log; log(){ echo "$(date -u +%FT%TZ) $(cr) [$MODE] $*" | tee -a "$LOG"; }
prog(){ echo "- $(TZ=America/Costa_Rica date '+%H:%M CR'): activate_ff $MODE: $*" >> "$D/PROGRESS.md"; }
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
  local x; for x in $((c-1)) $c $((c+1)) $((c+2)); do   # every trained epoch decided by the gate (+ RED restore finished)
    [[ -e $E/$x/TRAINING-RESULT.json && ! -e $GE/epochs/$x/DECISION.json ]] && return 1
    [[ -e $GE/epochs/$x/DECISION.json ]] && grep -q '"decision": *"red"' $GE/epochs/$x/DECISION.json && [[ ! -e $GE/epochs/$x/TRAINER-RESTORED.json ]] && return 1
  done
  systemctl --user list-units --no-legend 'affine-gate-restore-*' | grep -q -E 'active|activating' && return 1
  [[ $(trainbusy) == 0 ]] || return 1
  N "find $POD/fp32-master -newermt '-75 seconds' | grep -q ." && return 1
  return 0; }
kill_helpers(){ # PID-scoped: v83_dp helper PIDs, cmdline verified, TERM then KILL those PIDs only
  N 'pids=""; for p in $(pgrep -f "[h]ost-v833[gh]/v83_dp.py"); do tr "\0" " " </proc/$p/cmdline | grep -q "host-v833[gh]/v83_dp.py" && pids="$pids $p"; done
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
    local line new; line=$(grep '^ExecStart="/usr/bin/ssh"' "$TD/$G_TC"); new=${line//$G_J --sha256 $G_S/$H_J --sha256 $H_S}
    [[ "$new" != "$line" && "$new" == *"$H_J --sha256 $H_S"* ]] || return 15
    printf '%s\n%s\n%s\n%s\n%s\n' "# MiMo trainer v8.3.3h (= v8.3.3g + free-file batches: miner rewards as given). Jacob pt898u 2026-10-30." \
      "# Installed by $AV/activate_ff.sh $(cr). Revert: MODE=revert bash $AV/activate_ff.sh" "[Service]" "ExecStart=" "$new" > "$AV/$H_TC"
    install -m 600 "$AV/$H_TC" "$TD/$H_TC"
  else rm -f "$TD/$H_TC"; fi
  systemctl --user daemon-reload
  grep -q "$tj --sha256 $ts" <(systemctl --user show $SUP -p ExecStart --value) || return 16
  systemctl --user start $SUP
  ok=0; for i in $(seq 1 60); do N "$OW --config $POD/$tj --sha256 $ts --status" | grep -q '"alive": true' && { ok=1; break; }; sleep 5; done
  [[ $ok == 1 ]] || return 17
  log "trainer -> $tj ($ts) alive"; return 0; }
diskok(){ local f; f=$(N "df -B1G --output=avail $POD | tail -1" | tr -d ' '); [[ $f =~ ^[0-9]+$ ]] && (( f >= NEED_GB )) && { log "pod free ${f} GB >= $NEED_GB"; return 0; }; log "pod free ${f:-?} GB < $NEED_GB"; return 1; }
jset(){ # FILE key=json ... : atomic JSON update (keeps mode 0600 for new files)
  $PY - "$@" <<'PY'
import json, os, sys
p = sys.argv[1]; d = json.load(open(p)) if os.path.exists(p) else {}
for kv in sys.argv[2:]:
    k, v = kv.split('=', 1)
    if v == 'DELETE': d.pop(k, None)
    else: d[k] = json.loads(v)
t = p + '.ff-new'; open(t, 'w').write(json.dumps(d, indent=1, sort_keys=True) + '\n'); os.chmod(t, os.stat(p).st_mode & 0o777 if os.path.exists(p) else 0o600); os.replace(t, p)
PY
}
gate_restart(){ local i; for i in $(seq 1 ${GWAIT:-180}); do [[ -e $GE/GATE-BUSY.json ]] || break; sleep 5; done
  [[ -e $GE/GATE-BUSY.json ]] && { log "gate still busy - restart deferred (run: systemctl --user restart affine-promotion-gate)"; prog "gate restart DEFERRED (busy)"; return 0; }
  systemctl --user restart affine-promotion-gate.service; log "affine-promotion-gate restarted"; }
services(){ # install|restore
  local name path unit sha
  while IFS='|' read -r name path unit sha; do
    if [[ $1 == install ]]; then
      cp -a "$path" "$B/svc-$name.py"
      install -m 644 "$D/stage/services/$name.py" "$path.ff-new"; mv -f "$path.ff-new" "$path"
    else
      [[ -f "$B/svc-$name.py" ]] || { log "no backup for $name"; continue; }
      cp -a "$B/svc-$name.py" "$path.ff-old"; mv -f "$path.ff-old" "$path"
    fi
    [[ $unit == affine-promotion-gate ]] && continue
    systemctl --user restart $unit.service; log "$unit restarted ($1)"
  done <<< "$SVC"
  gate_restart; }
stage(){
  (cd $REL && sha256sum --quiet -c SIGNED-MANIFEST.sha256) || die "release manifest"
  local name path unit sha
  while IFS='|' read -r name path unit sha; do
    [[ $(sha256sum < "$path" | cut -c1-64) == "$sha" ]] || die "live $name changed since the patch was made ($path)"
    $PY -m py_compile "$D/stage/services/$name.py" || die "compile $name"
  done <<< "$SVC"
  for f in ff_format ff_lifecycle ff_launcher ff_service; do $PY -m py_compile $R/$f.py || die "compile $f"; done
  tar cf - -C $REL trainer-host-v833h | N "set -e; cd $POD; if [ ! -d host-v833h ]; then rm -rf .host-v833h.new; mkdir -m 700 .host-v833h.new; tar --no-same-owner -xf - -C .host-v833h.new; chown -R root:root .host-v833h.new; mv .host-v833h.new/trainer-host-v833h host-v833h; rmdir .host-v833h.new; else cat >/dev/null; fi
    sed 's#/host-v833g/#/host-v833h/#g' $G_J > .$H_J.new; chmod 600 .$H_J.new; mv .$H_J.new $H_J"
  (cd $REL/trainer-host-v833h && find . -type f -print0 | sort -z | xargs -0 sha256sum) > $AV/host-v833h.sha256
  N "cd $POD/host-v833h && sha256sum --quiet -c -" < $AV/host-v833h.sha256 || die "pod host-v833h sha mismatch"
  N "cd $POD && diff <(sed 's#/host-v833h/#/host-v833g/#g' $H_J) $G_J >/dev/null && ! grep -q host-v833g $H_J" || die "worker config diff"
  N "cd $POD/source-final-v823u && env CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 MKL_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH=$POD/source-final-v823u TOKENIZERS_PARALLELISM=false timeout 240 $POD/runtime-exact/venv/bin/python -B $POD/host-v833h/mimo_host_trainer_v83.py --operator-config $POD/host-v833h/TRAINER-HOST.v8.3.3.ROOT-ADMITTED.private.json --check -- >/dev/null 2>&1" || die "v833h --check failed (admission/pins/install)"
  H_S=$(N "sha256sum < $POD/$H_J" | cut -d' ' -f1); [[ $H_S =~ ^[0-9a-f]{64}$ ]] || die "H_S"; echo "$H_S" > $AV/H_S
  log "staged host-v833h + $H_J ($H_S)"
  $PY - "$R" "$L/mimo_v829_train_gate_launcher.py" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
r, l = Path(sys.argv[1]), sys.argv[2]
files = {n: hashlib.sha256((r / n).read_bytes()).hexdigest() for n in ('ff_launcher.py', 'ff_lifecycle.py', 'ff_format.py')}
t = r / '.FF-PINS.json.new'; t.write_text(json.dumps(dict(version='sn120-ff-pins-v1', v829_launcher=l, files=files), indent=1) + '\n')
os.chmod(t, 0o600); os.replace(t, r / 'FF-PINS.json')
PY
  [[ -f $FF ]] || $PY -c "
import json,os;p='$FF';d=dict(enabled=False,from_epoch=0,legacy_cutoff_unix=$CUTOFF,legacy_capture=True,tokenizer_dir='$TOK',authorized='Jacob 2026-10-30 pt898u')
open(p,'w').write(json.dumps(d,indent=1)+'\n');os.chmod(p,0o600)"
  mkdir -p -m 755 $WEB/ff; install -m 644 $R/affine_batch.py $WEB/ff/affine_batch.py
  cat > $AV/affine-ff.service <<UNIT
[Unit]
Description=SN120 free-file: private upload links + batch-file readiness (RA/20261030-free-file)
After=network-online.target
[Service]
Type=simple
WorkingDirectory=$D
ExecStart=$PY -u -B $R/ff_service.py
Restart=always
RestartSec=20
[Install]
WantedBy=default.target
UNIT
  install -m 644 $AV/affine-ff.service $UNITS/affine-ff.service; systemctl --user daemon-reload
  systemctl --user restart affine-ff.service; sleep 3; systemctl --user is-active --quiet affine-ff.service || die "affine-ff not active"
  log "staged: FF-PINS, FF.json (enabled $(grep -o '"enabled": [a-z]*' $FF)), affine-ff.service active, website/ff/affine_batch.py"; }
case $MODE in
stage) stage; prog "staged (trainer v833h on pod, FF-PINS, affine-ff service)"; log done ;;
switch)
  [[ -f $TD/$H_TC ]] && die "v833h drop-in already installed"
  grep -q "$G_J --sha256 $G_S" <(systemctl --user show $SUP -p ExecStart --value) || die "live trainer is not v833g (rebase needed)"
  grep -q "^ExecStart=$PY -B $L/mimo_v829_train_gate_launcher.py" "$LD/zzzz~~~~~~~~~~~~~v829-train-gate.conf" || die "lifecycle drop-in changed"
  [[ $(ls $LD | tail -1) == 'zzzz~~~~~~~~~~~~~v829-train-gate.conf' ]] || die "unexpected newer lifecycle drop-in"
  diskok || die "pod disk precheck failed"
  stage
  B=$AV/backup-$(date -u +%Y%m%dT%H%M%SZ); mkdir -m 700 $B; echo $B > $AV/LAST-BACKUP
  cp -a $TD $B/trainer-dropins; cp -a $LD $B/lifecycle-dropins; cp -a $PG $B/PROMOTION-GATE.json; cp -a $FF $B/FF.json
  systemctl --user show $SUP -p ExecStart --value > $B/trainer-ExecStart.txt
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
  log "trainer idle (epoch $(cur)), lifecycle stopped + locks held - switching"
  H_S=$(cat $AV/H_S); rc=0; switch_trainer $G_J $G_S $H_J $H_S install || rc=$?
  if [[ $rc != 0 ]]; then log "v833h switch failed rc=$rc - restoring v833g"; rm -f $TD/$H_TC; systemctl --user daemon-reload
    switch_trainer $H_J $H_S $G_J $G_S remove || log "WARNING restore rc=$?"; systemctl --user start affine-mimo-lifecycle.service; die "trainer switch failed rc=$rc (v833g restored)"; fi
  W=$(cur); while [[ -e $E/$W/TRAINING-JOB.json ]]; do W=$((W+1)); done      # first epoch whose turn is selected by ff_launcher
  jset $FF enabled=true from_epoch=$W "switched_at_cr=\"$(cr)\""
  jset $PG turn_batch_counted_from_epoch=$W
  services install
  printf '%s\n%s\n%s\n%s\n%s\n' "# SN120 free-file lifecycle (v8.2.9x launcher + release/ff_launcher.py). Jacob pt898u 2026-10-30." \
    "# Installed by $AV/activate_ff.sh $(cr). Revert: MODE=revert bash $AV/activate_ff.sh" "[Service]" "ExecStart=" \
    "ExecStart=$PY -B $R/ff_launcher.py $L_ARGS" > $AV/$L_TC
  install -m 600 $AV/$L_TC "$LD/$L_TC"; systemctl --user daemon-reload
  systemctl --user show affine-mimo-lifecycle.service -p ExecStart --value | grep -q "$R/ff_launcher.py" || die "lifecycle ExecStart not ff_launcher (run MODE=revert)"
  systemctl --user start affine-mimo-lifecycle.service; exec 8>&- 9>&-; log "lifecycle started on ff_launcher, locks released"
  echo "{\"from_epoch\": $W, \"trainer\": \"v833h\", \"sha256\": \"$H_S\", \"at\": \"$(cr)\", \"backup\": \"$B\"}" > $AV/ACTIVE.json
  systemctl --user reset-failed ff-watchdog.service 2>/dev/null || true
  systemd-run --user --unit=ff-watchdog --collect -p Restart=on-failure --setenv=FROM=$W --setenv=WATCH_UNTIL=$((CUTOFF+6*3600)) $PY -u -B $AV/watchdog_ff.py
  prog "LIVE free-file from epoch $W (trainer v833h $H_S); ff-watchdog started"; log done ;;
restart)   # reload release/*.py into the lifecycle (same ExecStart): waits for an idle trainer, stops/starts with both locks held
  systemctl --user show affine-mimo-lifecycle.service -p ExecStart --value | grep -q "$R/ff_launcher.py" || die "lifecycle is not on ff_launcher"
  ok=0; for i in $(seq 1 ${IWAIT:-720}); do
    if idle; then exec 8<>$STATE/lifecycle.lock; exec 9>$RA/LIFECYCLE-ACTIVATION.lock
      if flock -w 30 9 && flock -w 30 8; then systemctl --user stop affine-mimo-lifecycle.service; c=$(cur)
        if ! openjob $c && [[ $(trainbusy) == 0 ]]; then ok=1; break; fi
        systemctl --user start affine-mimo-lifecycle.service; fi; exec 8>&- 9>&-; fi
    sleep 5; done
  [[ $ok == 1 ]] || die "trainer never idle"
  systemctl --user start affine-mimo-lifecycle.service; exec 8>&- 9>&-
  log "lifecycle restarted at epoch $(cur) (${REASON:-reload})"; prog "lifecycle restarted at epoch $(cur) (${REASON:-reload})"; log done ;;
revert)
  B=$(cat $AV/LAST-BACKUP); [[ -d $B ]] || die "no backup dir"
  [[ ${FROM_WATCHDOG:-0} == 1 ]] || systemctl --user stop ff-watchdog.service 2>/dev/null || true
  jset $FF enabled=false "reverted_at_cr=\"$(cr)\"" "revert_reason=\"${REASON:-manual}\""     # ff_lifecycle inert from here on
  if [[ ${FORCE:-0} != 1 ]]; then for i in $(seq 1 ${IWAIT:-720}); do idle && break; sleep 5; done; fi
  exec 9>$RA/LIFECYCLE-ACTIVATION.lock; flock -w 120 9 || true; exec 8<>$STATE/lifecycle.lock; flock -w 60 8 || true
  systemctl --user stop affine-mimo-lifecycle.service
  rm -f "$LD/$L_TC"; systemctl --user daemon-reload
  rc=0; if [[ -f $TD/$H_TC ]]; then H_S=$(cat $AV/H_S); switch_trainer $H_J $H_S $G_J $G_S remove || rc=$?; fi
  systemctl --user start affine-mimo-lifecycle.service; exec 8>&- 9>&-
  jset $PG turn_batch_counted_from_epoch=DELETE
  services restore
  rm -f $AV/ACTIVE.json; [[ $rc == 0 ]] || die "trainer revert failed rc=$rc (lifecycle + sidecars restored)"
  prog "REVERTED to v8.2.9x launcher + trainer v8.3.3g (${REASON:-manual})"; log done ;;
esac
