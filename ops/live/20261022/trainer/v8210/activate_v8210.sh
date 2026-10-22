#!/usr/bin/env bash
# MiMo trainer host v8.2.10: v8.2.5 + engine.optimizer_persistence='none' (optimizer state stays in trainer memory;
# the epoch report never waits on writing it). Authorized by Jacob 2026-10-22 ("saving the optimizer state should not
# be important can we remove this requirement"). Lifecycle / trainer_config / policy / closure / writer / scorer UNCHANGED;
# lifecycle NOT restarted. Pattern = activate_v825.sh: new pod dir (nothing overwritten), owned-worker --check, NEW
# drop-in, drain-restart ONLY while no TRAINING-JOB is in flight. No chain tx, no messages, no pods rented/released.
# Usage: activate_v8210.sh --preflight | --activate | --arm <epoch_index whose TRAINING-RESULT to wait for>
set -euo pipefail
umask 077
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
D=/home/const/subnet120-rewrite/state/root-audits/20261022-async-optimizer-save
SG=$D/release-v8210-signed; A=$D/activation-v8210
STATE=/home/const/subnet120-rewrite/state/mimo-mainnet-v1
UNITS=/home/const/.config/systemd/user
PY=/home/const/subnet120-rewrite/.venv/bin/python
LIVE_DROPIN="zzzz~~~~~~~pod-20261022-owned-worker.conf"
TC="zzzz~~~~~~~~v8210s-owned-worker.conf"
RSSH=/home/const/subnet120-rewrite/state/root-audits/20261022-trainer-pod-migration/rssh.sh
POD=/workspace/affine-mimo-trainer-prod
V825_SHA=8d796a925cb71981c4339b0cef710885b3925060ed7023cb639a2bd1670484a9
MODE=${1:---preflight}
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$D/activate-v8210.log"; }
die(){ log "ABORT: $*"; exit 1; }
# pod sshd can refuse new logins under disk load: retry
P(){ local i; for i in 1 2 3 4 5 6; do "$RSSH" "$@" && return 0; rc=$?; [[ $rc == 255 ]] || return $rc; sleep 10; done; return 255; }
cur(){ "$PY" -c "import json;print(json.load(open('$STATE/CURRENT.json'))['$1'])"; }
inflight(){ local e; e=$(cur epoch_index); [[ -e "$STATE/epochs/$e/TRAINING-JOB.json" && ! -e "$STATE/epochs/$e/TRAINING-RESULT.json" ]]; }
log "v8.2.10 activation mode=$MODE"
(cd "$SG" && sha256sum --quiet -c SIGNED-MANIFEST.sha256) || die "release-v8210-signed manifest mismatch"
if [[ "$MODE" == --arm ]]; then
  WAIT_E=${2:?epoch index}; log "armed: waiting for epochs/$WAIT_E/TRAINING-RESULT.json (then activate before the next dispatch)"
  for i in $(seq 1 1080); do   # <= 6 h
    [[ -e "$STATE/epochs/$WAIT_E/TRAINING-RESULT.json" ]] && break
    [[ -e "$STATE/epochs/$WAIT_E/RECOVERY-REQUIRED.json" ]] && die "e$WAIT_E RECOVERY-REQUIRED; not activating (operator decision)"
    sleep 20
  done
  [[ -e "$STATE/epochs/$WAIT_E/TRAINING-RESULT.json" ]] || die "e$WAIT_E TRAINING-RESULT did not appear within 6 h"
  log "e$WAIT_E TRAINING-RESULT present ($(stat -c %y "$STATE/epochs/$WAIT_E/TRAINING-RESULT.json")); activating"
  sleep 30   # let the worker finish its post-report bookkeeping
  exec "$0" --activate
fi
EI=$(cur epoch_index); log "CURRENT epoch_index=$EI round=$(cur round)"
inflight && die "e$EI TRAINING-JOB in flight; never restart the trainer mid-job (target the next epoch)"
OW825=(P "$POD/runtime-exact/venv/bin/python -I -B $POD/owned-worker-v7.py --config $POD/owned-worker-v825s.json --sha256 $V825_SHA")
mkdir -p -m 700 "$A/pod" "$A/trainer"
if [[ ! -e "$A/pod/owned-worker-v8210s.json" ]]; then
  P "cat $POD/owned-worker-v825s.json" > "$A/pod/owned-worker-v825s.json.from-pod"
  [[ "$(sha256sum "$A/pod/owned-worker-v825s.json.from-pod" | cut -c1-64)" == "$V825_SHA" ]] || die "pod v825s config digest"
  "$PY" - "$A" <<'PY'
import json, sys
from pathlib import Path
A = Path(sys.argv[1]); raw = (A/'pod/owned-worker-v825s.json.from-pod').read_text()
base = json.loads(raw); POD = '/workspace/affine-mimo-trainer-prod'
cmd = list(base['command'])
assert cmd[2] == POD + '/host-v825/mimo_host_trainer_v825.py' and cmd[4] == POD + '/host-v825/TRAINER-HOST.v8.2.5.ROOT-ADMITTED.private.json'
style = json.dumps(base, indent=1, sort_keys=True) + '\n'
for name, cfg in (('owned-worker-v8210s.json', 'TRAINER-HOST.v8.2.10.ROOT-ADMITTED.private.json'),
                  ('owned-worker-v8210s-sync.json', 'TRAINER-HOST.v8.2.10-sync.ROOT-ADMITTED.private.json')):
    c = list(cmd); c[2] = POD + '/host-v8210/mimo_host_trainer_v8210.py'; c[4] = POD + '/host-v8210/' + cfg
    new = dict(base, command=c)
    assert {k for k in base if base[k] != new[k]} == {'command'}
    (A/'pod'/name).write_text((json.dumps(new, indent=1, sort_keys=True) + '\n') if style == raw else json.dumps(new, sort_keys=True))
PY
  chmod 600 "$A"/pod/*
fi
NEW_SHA=$(sha256sum "$A/pod/owned-worker-v8210s.json" | cut -c1-64); SYNC_SHA=$(sha256sum "$A/pod/owned-worker-v8210s-sync.json" | cut -c1-64)
OW8210=(P "$POD/runtime-exact/venv/bin/python -I -B $POD/owned-worker-v7.py --config $POD/owned-worker-v8210s.json --sha256 $NEW_SHA")
ULIVE="$UNITS/affine-mimo-trainer-supervised.service.d/$LIVE_DROPIN"
grep -q "owned-worker-v825s.json --sha256 $V825_SHA" "$ULIVE" || die "live trainer drop-in is not v825s"
grep -q "150.136.210.177" "$ULIVE" || die "live trainer drop-in is not eager-fox-7b"
if [[ ! -e "$A/trainer/$TC" ]]; then
  line=$(grep '^ExecStart="/usr/bin/ssh"' "$ULIVE"); new=${line//owned-worker-v825s.json --sha256 $V825_SHA/owned-worker-v8210s.json --sha256 $NEW_SHA}
  [[ "$new" != "$line" ]] || die "ExecStart rewrite"
  printf '%s\n%s\n%s\n%s\n%s\n' "# MiMo v8.2.10: trainer owned worker = v8.2.5 host layer + engine.optimizer_persistence=none (optimizer state stays in memory; report never waits on it)." \
    "# Installed by activate_v8210.sh 2026-10-22 (authorized by Jacob). Same pod (eager-fox-7b) as $LIVE_DROPIN. Rollback: see $D/ROLLBACK.md" "[Service]" "ExecStart=" "$new" > "$A/trainer/$TC"
  chmod 600 "$A/trainer/$TC"
fi
log "owned-worker-v8210s sha256=$NEW_SHA (sync variant $SYNC_SHA)"
if ! P "test -e $POD/host-v8210"; then
  P "mkdir -m 700 $POD/host-v8210"
  (cd "$SG/trainer-host-v8210" && tar -cf - .) | P "tar --no-same-owner -C $POD/host-v8210 -xf - && chmod -R go-rwx $POD/host-v8210"
fi
P "test -e $POD/owned-worker-v8210s.json || (umask 077 && cat > $POD/owned-worker-v8210s.json)" < "$A/pod/owned-worker-v8210s.json"
P "test -e $POD/owned-worker-v8210s-sync.json || (umask 077 && cat > $POD/owned-worker-v8210s-sync.json)" < "$A/pod/owned-worker-v8210s-sync.json"
(cd "$SG/trainer-host-v8210" && sha256sum $(find . -type f | sort)) > "$A/pod-host-files.sha256"
P "cd $POD/host-v8210 && sha256sum -c --quiet" < "$A/pod-host-files.sha256" || die "pod host files digest"
[[ "$(P "sha256sum $POD/owned-worker-v8210s.json" | cut -c1-64)" == "$NEW_SHA" ]] || die "pod owned-worker-v8210s digest"
[[ "$(P "sha256sum $POD/owned-worker-v8210s-sync.json" | cut -c1-64)" == "$SYNC_SHA" ]] || die "pod owned-worker-v8210s-sync digest"
log "pod files verified; owned-worker --check (launcher admission + original-method pins on the pod source; no model, no claims)"
"${OW8210[@]}" --check 2>&1 | tail -3 | tee -a "$D/activate-v8210.log" | grep -q '"check": true' || die "owned-worker v8210s --check failed (nothing restarted)"
P "$POD/runtime-exact/venv/bin/python -I -B $POD/owned-worker-v7.py --config $POD/owned-worker-v8210s-sync.json --sha256 $SYNC_SHA --check" 2>&1 | tail -3 | tee -a "$D/activate-v8210.log" | grep -q '"check": true' || die "sync-variant --check failed (nothing restarted)"
[[ "$MODE" == --activate ]] || { log "preflight OK; nothing restarted"; exit 0; }
TS=$(date -u +%Y%m%dT%H%M%SZ); BK=$D/backup-trainer-$TS; mkdir -m 700 "$BK"
tar -C "$UNITS" -cf "$BK/systemd-user-units.tar" affine-mimo-trainer-supervised.service.d affine-mimo-trainer-tunnel.service.d
cp -a "$STATE/CURRENT.json" "$BK/CURRENT.json.pre"
systemctl --user show affine-mimo-trainer-supervised.service -p ExecStart --value > "$BK/trainer-execstart.pre.txt"
"${OW825[@]}" --status > "$BK/worker-status.pre.json" 2>&1 || true
P "cat $POD/trainer-state/affine-mimo9b-grpo-20261013/CURRENT.json" > "$BK/pod-trainer-CURRENT.json.pre" 2>/dev/null || true
sha256sum "$BK"/* > "$BK/BACKUP.sha256"; log "backups -> $BK"
inflight && die "job appeared; nothing changed"
install -m 600 "$A/trainer/$TC" "$UNITS/affine-mimo-trainer-supervised.service.d/$TC"
systemctl --user daemon-reload
if inflight; then rm -f "$UNITS/affine-mimo-trainer-supervised.service.d/$TC"; systemctl --user daemon-reload; die "job appeared; drop-in removed, nothing restarted"; fi
grep -q "owned-worker-v8210s.json --sha256 $NEW_SHA" <(systemctl --user show affine-mimo-trainer-supervised.service -p ExecStart --value) || die "effective ExecStart is not v8210s (drop-in order)"
systemctl --user stop affine-mimo-trainer-supervised.service
"${OW825[@]}" --stop-owned-after-drain | tee -a "$D/activate-v8210.log" || log "note: no live worker"
st=True; for i in $(seq 1 120); do st=$("${OW825[@]}" --status | "$PY" -c "import json,sys;print(json.load(sys.stdin)['alive'])") || st=True; [[ "$st" == False ]] && break; sleep 5; done
[[ "$st" == False ]] || die "v825 worker still alive after 10 min (watcher left stopped)"
inflight && log "WARNING: a TRAINING-JOB appeared during drain (it will be claimed by the v8210 worker)"
systemctl --user start affine-mimo-trainer-supervised.service
ok=0; for i in $(seq 1 60); do "${OW8210[@]}" --status | grep -q '"alive": true' && { ok=1; break; }; sleep 5; done
[[ $ok == 1 ]] || die "v8210 worker not alive (run ROLLBACK.md steps)"
sleep 20
P "grep -a 'trainer-host-v8210' $POD/owned-worker-v7.log | tail -1 | cut -c1-600" | tee -a "$D/activate-v8210.log"
log "ACTIVATED trainer v8.2.10 optimizer_persistence=none (cold optimizer restore of the e$((EI-1))/current generation at the next job, legacy full state -> exact moments)"
