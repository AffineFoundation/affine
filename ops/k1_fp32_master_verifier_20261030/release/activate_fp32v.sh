#!/usr/bin/env bash
# RA/20261030-fp32-verifier activation (Jacob t831u/pt831u "push the changes ahorita").
#   MODE=switch           WAIT_EPOCH=E: after gate DECISION of e(E): (1) verifier = new probe_worker.py on the probe pod + close the open
#                          held-out round (fresh round, same head, at the next sample); (2) trainer v8.3.3u -> v8.3.3f at the e(E)/e(E+1)
#                          boundary (lifecycle stopped <= a few min, after any RED restore); (3) start publisher + watchdog.
#                          SKIP_VERIFIER=1 / SKIP_TRAINER=1 to do one part only.
#   MODE=revert-trainer   v8.3.3f -> v8.3.3u at the next boundary (or now if the trainer is idle and no train job exists).
#   MODE=revert-verifier  restore probe_worker.py backup right after the next gate DECISION + close the open round (re-baseline).
#   MODE=revert           both.
set -euo pipefail; umask 077
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
MODE=${MODE:?MODE=switch|revert|revert-trainer|revert-verifier}
RA=/home/const/subnet120-rewrite/state/root-audits; D=$RA/20261030-fp32-verifier; AV=$D/activation; mkdir -p -m 700 $AV
STATE=/home/const/subnet120-rewrite/state/mimo-mainnet-v1; E=$STATE/epochs; GV=$RA/20261026-promotion-gate; GE=$GV/state/epochs
PY=/home/const/subnet120-rewrite/.venv/bin/python; SVC=affine-mimo-lifecycle.service
UNITS=/home/const/.config/systemd/user; SUP=affine-mimo-trainer-supervised.service; TD=$UNITS/$SUP.d
U_TC='zzzz~~~~~~~~~~~~~~v833u-owned-worker.conf'; F_TC='zzzz~~~~~~~~~~~~~~~v833f-owned-worker.conf'
U_J=owned-worker-v833u.json; U_S=c7a644df8f0efc2dc172fecb9479904fd879c41e276ca78cbdb3b8d17cdb39b6
F_J=owned-worker-v833f.json; F_S=e8d6c843e05d0b0df7d59d07411fa789d04e411307524dd787acc3b14de16818
POD=/workspace/affine-mimo-trainer-prod; OW="$POD/runtime-exact/venv/bin/python -I -B $POD/owned-worker-v7.py"
TKH=/home/const/.lium/known_hosts/ef2b6085-adae-43e2-8a13-39182a3b3f7a
PKH=/home/const/.lium/known_hosts/b4f3da9d-df66-436a-af38-829d1606966d; PODD=/root/affine-env-probe
PW_NEW=$D/stage/probe_worker.py; PW_NEW_S=$(sha256sum < $PW_NEW | cut -d' ' -f1)
SSHO=(-T -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10 -o ServerAliveInterval=20 -o ServerAliveCountMax=6)
N(){ local i rc; for i in 1 2 3 4 5 6; do ssh "${SSHO[@]}" -o UserKnownHostsFile=$TKH -p ${TRAINER_POD_PORT:?} root@${TRAINER_POD_HOST:?} "$@" && return 0; rc=$?; [[ $rc == 255 ]] || return $rc; sleep 5; done; return 255; }
PP(){ local i rc; for i in 1 2 3 4 5 6; do ssh "${SSHO[@]}" -o UserKnownHostsFile=$PKH -p ${PROBE_POD_PORT:?} root@${PROBE_POD_HOST:?} "$@" && return 0; rc=$?; [[ $rc == 255 ]] || return $rc; sleep 3; done; return 255; }
cr(){ TZ=America/Costa_Rica date "+%H:%M:%S CR"; }
LOG=$AV/activate.log; log(){ echo "$(date -u +%FT%TZ) $(cr) [$MODE] $*" | tee -a "$LOG"; }
prog(){ echo "- $(TZ=America/Costa_Rica date '+%H:%M CR'): activate_fp32v $MODE: $*" >> "$D/PROGRESS.md"; }
die(){ log "ABORT: $*"; prog "ABORT: $*"; exit 1; }
cur(){ $PY -c "import json;print(json.load(open('$STATE/CURRENT.json'))['epoch_index'])"; }
trainjobs(){ # same query as gate trainer_restore.sh; bounded (the 51 GB roles DB is often write-locked). err -> caller relies on file evidence
  timeout 12 $PY -c "
import sqlite3;c=sqlite3.connect('file:$STATE/roles.sqlite3?mode=ro',uri=True,timeout=5)
print(c.execute(\"select count(*) from jobs where role='train' and status in ('queued','leased')\").fetchone()[0])" 2>/dev/null || echo err; }
trainer_alive(){ N "$OW --config $POD/$1 --sha256 $2 --status" 2>/dev/null | $PY -c 'import json,sys;d=json.load(sys.stdin);print(bool(d.get("alive")) and (d.get("identity") or {}).get("config_sha256")==sys.argv[1])' "$2" 2>/dev/null || echo False; }
close_round(){ # REASON EPOCH -> closes the open held-out round (file API of heldout.Rounds; the gate opens a fresh round at the next sample)
  $PY - "$1" "$2" <<'PYX'
import sys, json, os, time
sys.path.insert(0, '/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate/code')
from pathlib import Path
import heldout as HO
def cr(ts=None):
    ts = time.time() if ts is None else ts; return time.strftime('%Y-%m-%d %H:%M:%S CR (UTC-6)', time.gmtime(ts - 6 * 3600))
def wj(path, obj, mode=0o600):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    t = path.with_name('.' + path.name + '.tmp'); fd = os.open(t, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    with os.fdopen(fd, 'w') as f: f.write(json.dumps(obj, indent=1, sort_keys=True) + '\n')
    os.chmod(t, mode); os.replace(t, path)
def rj(path, default=None):
    try: return json.loads(Path(path).read_text())
    except Exception: return default
def log(**k):
    k['at_cr'] = cr(); k['by'] = 'fp32-verifier activate'
    open('/home/const/subnet120-rewrite/state/root-audits/20261030-fp32-verifier/activation/rounds.log', 'a').write(json.dumps(k) + '\n')
R = HO.Rounds(Path('/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate/state/heldout'), wj, rj, log, cr)
c = R.current()
if not c: print('no-open-round'); sys.exit(0)
print('closed' if R.close(c['id'], sys.argv[1], int(sys.argv[2])) else 'already-closed', c['id'])
PYX
}
install_pw(){ # FILE EXPECTED_SHA LABEL
  local f=$1 s=$2 ts; ts=$(date -u +%Y%m%dT%H%M%SZ)
  PP "cd $PODD && cp -p probe_worker.py probe_worker.py.bak-pre-$3-$ts && cat > .probe_worker.py.new && test \"\$(sha256sum < .probe_worker.py.new | cut -d' ' -f1)\" = $s && chmod --reference=probe_worker.py .probe_worker.py.new && mv .probe_worker.py.new probe_worker.py && sha256sum probe_worker.py" < "$f" | tee -a "$LOG"
  [[ $(PP "sha256sum < $PODD/probe_worker.py" | cut -d' ' -f1) == "$s" ]] || return 1
  echo "$PODD/probe_worker.py.bak-pre-$3-$ts" > $AV/PW-BACKUP.$3; }
gate_idle(){ # no H/C probe run active on the pod for the gate (status.json state of the last epochs)
  $PY - <<PYX
import json; s=json.load(open('$GV/state/status.json'))['state']
bad=[k+':'+str(v) for k,v in s.items() if not k.startswith('restore') and k!='keep' and v not in ('decided','no-sample','no-candidate','skip','heldout-round-closed',None)]
print('idle' if not bad else ' '.join(bad))
PYX
}
wait_decision(){ local e=$1 t0; t0=$(date +%s); while [[ ! -f $GE/$e/DECISION.json ]]; do sleep 0.3; (( $(date +%s)-t0 < ${MAXWAIT:-10800} )) || return 1; done; }
boundary(){ # NEXT: wait CURRENT==NEXT, prev FINALIZED, no TRAINING-JOB in NEXT; lock + stop lifecycle. fd 8 holds the lock.
  local nx=$1 t0; t0=$(date +%s)
  while true; do
    (( $(date +%s)-t0 < ${BWAIT:-5400} )) || return 1
    if [[ $(cur) == $nx && -f $E/$((nx-1))/FINALIZED.json && ! -e $E/$nx/TRAINING-JOB.json ]]; then
      exec 8<>$STATE/lifecycle.lock; flock -w 30 8 || { exec 8>&-; sleep 1; continue; }
      systemctl --user stop $SVC
      if [[ $(cur) == $nx && ! -e $E/$nx/TRAINING-JOB.json && $(trainjobs) =~ ^(0|err)$ ]]; then log "boundary e$((nx-1))/e$nx: lifecycle stopped (lock held)"; return 0; fi
      systemctl --user start $SVC; exec 8>&-; log "boundary e$nx: job appeared - not here"; return 2
    fi
    [[ -e $E/$nx/TRAINING-JOB.json ]] && return 2
    sleep 0.5
  done; }
release_lifecycle(){ systemctl --user start $SVC; exec 8>&- || true; log "lifecycle started, lock released"; }
wait_restore(){ local e=$1 t0; t0=$(date +%s)
  [[ $($PY -c "import json;print(json.load(open('$GE/$e/DECISION.json')).get('decision'))") == red ]] || return 0
  while [[ ! -f $GE/$e/TRAINER-RESTORED.json ]]; do sleep 2; (( $(date +%s)-t0 < 900 )) || return 1; done
  sleep 3; while systemctl --user list-units --no-legend 'affine-gate-restore-*' | grep -q -E 'active|activating'; do sleep 2; done; log "e$e RED restore done"; }
switch_trainer(){ # FROM_J FROM_S TO_J TO_S install|remove
  local fj=$1 fs=$2 tj=$3 ts=$4 act=$5 i ok
  systemctl --user stop $SUP
  N "$OW --config $POD/$fj --sha256 $fs --stop-owned-after-drain" 2>&1 | cut -c1-300 | tee -a "$LOG" || true
  ok=0; for i in $(seq 1 60); do N "$OW --config $POD/$fj --sha256 $fs --status" | grep -q '"alive": false' && { ok=1; break; }; sleep 5; done
  [[ $ok == 1 ]] || return 11
  N "pkill -TERM -f '[h]ost-v833[a-z]*/v83_dp.py' || true; sleep 5; pkill -KILL -f '[h]ost-v833[a-z]*/v83_dp.py' || true"
  for i in $(seq 1 12); do N "ss -ltn | grep -q ':29583 '" || break; sleep 5; done
  N "ss -ltn | grep -q ':29583 '" && return 12
  if [[ $act == install ]]; then
    local line new; line=$(grep '^ExecStart="/usr/bin/ssh"' "$TD/$U_TC"); new=${line//$U_J --sha256 $U_S/$F_J --sha256 $F_S}
    [[ "$new" != "$line" && "$new" == *"$F_J --sha256 $F_S"* ]] || return 15
    printf '%s\n%s\n%s\n%s\n%s\n' "# MiMo trainer v8.3.3f (= v8.3.3u + persistent fp32 head master, engine.sgd_fp32_master) on golden-raven-7a. Jacob t831u 2026-10-30." \
      "# Installed by $D/activation/activate_fp32v.sh $(cr). Revert: MODE=revert-trainer bash $AV/activate_fp32v.sh (see $D/REVERT.md)." "[Service]" "ExecStart=" "$new" > "$AV/$F_TC"
    install -m 600 "$AV/$F_TC" "$TD/$F_TC"
  else rm -f "$TD/$F_TC"; fi
  systemctl --user daemon-reload
  grep -q "$tj --sha256 $ts" <(systemctl --user show $SUP -p ExecStart --value) || return 16
  systemctl --user start $SUP
  ok=0; for i in $(seq 1 60); do N "$OW --config $POD/$tj --sha256 $ts --status" | grep -q '"alive": true' && { ok=1; break; }; sleep 5; done
  [[ $ok == 1 ]] || return 17
  log "trainer -> $tj ($ts) alive"; return 0; }
do_trainer(){ # TO=f|u   E (decided epoch before the boundary)
  local to=$1 e=$2 rc nx=$(( $2 + 1 ))
  local tries=0
  while (( tries < ${BOUNDARY_TRIES:-3} )); do
    tries=$((tries+1)); rc=0; boundary $nx || rc=$?
    if [[ $rc == 0 ]]; then
      wait_restore $e || { release_lifecycle; die "RED restore of e$e not finished in 15 min (trainer unchanged)"; }
      rc=0
      if [[ $to == f ]]; then switch_trainer $U_J $U_S $F_J $F_S install || rc=$?
        if [[ $rc != 0 ]]; then log "v833f switch failed rc=$rc - restoring v833u"; switch_trainer $F_J $F_S $U_J $U_S remove || log "WARNING restore rc=$?"; release_lifecycle; die "trainer switch failed rc=$rc (v833u restored)"; fi
      else switch_trainer $F_J $F_S $U_J $U_S remove || rc=$?
        [[ $rc == 0 ]] || { release_lifecycle; die "trainer revert failed rc=$rc"; }
      fi
      release_lifecycle; echo "{\"trainer\": \"v833$to\", \"from_epoch\": $nx, \"at\": \"$(cr)\"}" > $AV/TRAINER-$to.json; return 0
    fi
    [[ $rc == 1 ]] && die "no boundary e$nx within wait"
    log "boundary e$nx missed (training job already issued) - waiting for e$nx DECISION and trying e$((nx+1))"
    wait_decision $nx || die "no e$nx decision"; e=$nx; nx=$((nx+1))
  done; die "no usable boundary after $tries tries"; }
case $MODE in
switch)
  WAIT_EPOCH=${WAIT_EPOCH:?}; (cd $D/release-v833f-signed && sha256sum --quiet -c SIGNED-MANIFEST.sha256) || die "release manifest"
  [[ $(N "sha256sum < $POD/$F_J" | cut -d' ' -f1) == $F_S ]] || die "v833f config not staged on pod"
  [[ $(N "sha256sum < $POD/host-v833f/mimo_host_trainer_v83.py" | cut -d' ' -f1) == dde1eeb1d510b3dbad7963f8d68190bd6fad8a5a3187bb5d694c74143dcb34cf ]] || die "launcher not staged"
  grep -q -E '^(HOLD|I CLOSE)' $RA/20261030-heldout-gate/NOTES-from-trainer-worker.md && HOLDNOTE=1 || HOLDNOTE=0
  [[ $HOLDNOTE == 0 ]] || die "held-out worker wrote HOLD/I CLOSE in the notes - not acting"
  log "armed: waiting for gate DECISION e$WAIT_EPOCH (probe_worker new $PW_NEW_S)"; prog "armed, waiting for e$WAIT_EPOCH DECISION"
  wait_decision $WAIT_EPOCH || die "no DECISION e$WAIT_EPOCH"
  log "e$WAIT_EPOCH DECISION: $($PY -c "import json;d=json.load(open('$GE/$WAIT_EPOCH/DECISION.json'));print(d.get('decision'), d.get('z', (d.get('evaluated') or {}).get('z')))")"
  if [[ ${SKIP_VERIFIER:-0} != 1 ]]; then
    gi=$(gate_idle); log "gate state at decision: $gi"
    install_pw $PW_NEW $PW_NEW_S verifierfix || die "probe_worker install failed"
    log "probe_worker -> $PW_NEW_S (backup $(cat $AV/PW-BACKUP.verifierfix))"
    r=$(close_round "grader change: Allan verifier fix 786cdb9 (probe_worker ${PW_NEW_S:0:8}) -> new round / re-baseline (RA/20261030-fp32-verifier)" $((WAIT_EPOCH+1)))
    log "held-out round: $r"; prog "VERIFIER LIVE: probe_worker ${PW_NEW_S:0:8} installed after e$WAIT_EPOCH DECISION; round $r; next sample opens a fresh round"
    echo "{\"verifier\": \"$PW_NEW_S\", \"from_epoch\": $((WAIT_EPOCH+1)), \"at\": \"$(cr)\", \"round\": \"$r\"}" > $AV/VERIFIER-ON.json
  fi
  systemctl --user is-active --quiet fp32v-watchdog || systemd-run --user --unit fp32v-watchdog --collect /usr/bin/env ${SKIP_TRAINER:+NO_TRAINER=1} FROM=$((WAIT_EPOCH+1)) $PY -u -B $D/release/watchdog_fp32v.py
  if [[ ${SKIP_TRAINER:-0} != 1 ]]; then
    do_trainer f $WAIT_EPOCH; prog "TRAINER LIVE: v8.3.3f from e$($PY -c "import json;print(json.load(open('$AV/TRAINER-f.json'))['from_epoch'])")"
    FROM=$($PY -c "import json;print(json.load(open('$AV/TRAINER-f.json'))['from_epoch'])")
    $PY -c "import json,os;p='$D/state/publisher.json';d=json.load(open(p)) if os.path.exists(p) else {};d['from_epoch']=$FROM;json.dump(d,open(p,'w'),indent=1)"
    systemctl --user is-active --quiet fp32-publisher || systemd-run --user --unit fp32-publisher --collect -p Restart=always -p RestartSec=30 $PY -u -B $D/release/fp32_publish.py
  fi
  log "switch done"; prog "switch done; fp32v-watchdog armed from e$((WAIT_EPOCH+1))";;
revert-trainer)
  e=$(cur); if [[ -f $GE/$e/DECISION.json || ! -f $E/$e/TRAINING-RESULT.json ]]; then :; fi
  # next boundary after the current epoch's decision (or right now if CURRENT has no training job yet)
  if [[ ! -e $E/$e/TRAINING-JOB.json ]]; then do_trainer u $((e-1)); else wait_decision $e; do_trainer u $e; fi
  prog "TRAINER REVERTED to v8.3.3u";;
revert-verifier)
  BK=$(cat $AV/PW-BACKUP.verifierfix); OLD_S=36d7f40235de555c
  e=$(cur); log "waiting for a gate DECISION (e$e) to swap back the probe worker"
  [[ ${NOW:-0} == 1 ]] || wait_decision $e
  PP "cd $PODD && sha256sum < $BK | cut -c1-16 | grep -q $OLD_S && cp -p probe_worker.py probe_worker.py.verifierfix-reverted-$(date -u +%H%M%S) && cp -p $BK probe_worker.py && sha256sum probe_worker.py" | tee -a "$LOG"
  r=$(close_round "grader revert: back to the live grader (probe_worker 36d7f402) -> re-baseline (RA/20261030-fp32-verifier)" $((e+1))); log "round: $r"
  prog "VERIFIER REVERTED (probe_worker 36d7f402), round $r";;
revert)
  MODE=revert-verifier NOW=${NOW:-0} bash $0 || true; MODE=revert-trainer bash $0;;
esac
