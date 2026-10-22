#!/usr/bin/env bash
# MiMo v8.2.5 trainer activation: v8.2.4 trainer host layer + engine.drop_unresolved=true (ONLY change).
# Lifecycle / trainer_config / policy / closure / writer / scorer UNCHANGED; lifecycle NOT restarted.
# Pattern = activate_v824.sh --trainer: new pod dir (nothing overwritten), owned-worker --check, drop-in, drain-restart
# ONLY while no TRAINING-JOB is in flight. No chain transactions, no messages, never touches the SUBNETOWNER mnemonic.
# Usage: activate_v825.sh [--preflight|--activate] [on|off]
set -euo pipefail
umask 077
D=/home/const/subnet120-rewrite/state/root-audits/20261021-v825-drop-unresolved
SG=$D/release-v825-signed; A=$D/activation-v825
STATE=/home/const/subnet120-rewrite/state/mimo-mainnet-v1
UNITS=/home/const/.config/systemd/user
PY=/home/const/subnet120-rewrite/.venv/bin/python
TC="zzzz~~~~~~v825s-owned-worker.conf"
POD_SSH=(ssh -T -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10 -o UserKnownHostsFile=/home/const/.lium/known_hosts/0a7aba3d-9443-445b-9601-c78ac0d14b1c -p 40301 root@89.169.120.92)
POD=/workspace/affine-mimo-trainer-prod
V824_SHA=b8ba9dca9687f7c4f7b3bc83d8be563af63ff1a74e7259f45791418a690357a4
MODE=${1:---preflight}
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$D/activate-v825.log"; }
die(){ log "ABORT: $*"; exit 1; }
cur(){ "$PY" -c "import json;print(json.load(open('$STATE/CURRENT.json'))['$1'])"; }
inflight(){ local e; e=$(cur epoch_index); [[ -e "$STATE/epochs/$e/TRAINING-JOB.json" && ! -e "$STATE/epochs/$e/TRAINING-RESULT.json" ]]; }
log "v8.2.5 activation mode=$MODE"
(cd "$SG" && sha256sum --quiet -c SIGNED-MANIFEST.sha256) || die "release-v825-signed manifest mismatch"
EI=$(cur epoch_index); log "CURRENT epoch_index=$EI round=$(cur round) max_tasks=$(cur max_tasks)"
inflight && die "e$EI TRAINING-JOB in flight; never restart the trainer mid-job (target the next epoch)"
OW824=("${POD_SSH[@]}" "$POD/runtime-exact/venv/bin/python -I -B $POD/owned-worker-v7.py --config $POD/owned-worker-v824s.json --sha256 $V824_SHA")
mkdir -p -m 700 "$A/pod" "$A/trainer"
if [[ ! -e "$A/pod/owned-worker-v825s.json" ]]; then
  "${POD_SSH[@]}" "cat $POD/owned-worker-v824s.json" > "$A/pod/owned-worker-v824s.json.from-pod"
  [[ "$(sha256sum "$A/pod/owned-worker-v824s.json.from-pod" | cut -c1-64)" == "$V824_SHA" ]] || die "pod v824s config digest"
  "$PY" - "$A" <<'PY'
import json, sys
from pathlib import Path
A = Path(sys.argv[1]); raw = (A/'pod/owned-worker-v824s.json.from-pod').read_text()
base = json.loads(raw); POD = '/workspace/affine-mimo-trainer-prod'
cmd = list(base['command'])
assert cmd[2] == POD + '/host-v824/mimo_host_trainer_v824.py' and cmd[4] == POD + '/host-v824/TRAINER-HOST.v8.2.4.ROOT-ADMITTED.private.json'
cmd[2] = POD + '/host-v825/mimo_host_trainer_v825.py'; cmd[4] = POD + '/host-v825/TRAINER-HOST.v8.2.5.ROOT-ADMITTED.private.json'
new = dict(base, command=cmd)
assert {k for k in base if base[k] != new[k]} == {'command'}
# same serialisation style as the v824s file (json indent=1 sort_keys? keep exact style by re-dumping like the original)
style = json.dumps(base, indent=1, sort_keys=True) + '\n'
out = (json.dumps(new, indent=1, sort_keys=True) + '\n') if style == raw else json.dumps(new, sort_keys=True)
(A/'pod/owned-worker-v825s.json').write_text(out)
off = dict(base, command=cmd[:4] + [POD + '/host-v825/TRAINER-HOST.v8.2.5-off.ROOT-ADMITTED.private.json'] + cmd[5:])
(A/'pod/owned-worker-v825s-off.json').write_text((json.dumps(off, indent=1, sort_keys=True) + '\n') if style == raw else json.dumps(off, sort_keys=True))
PY
  chmod 600 "$A"/pod/*
fi
NEW_SHA=$(sha256sum "$A/pod/owned-worker-v825s.json" | cut -c1-64); OFF_SHA=$(sha256sum "$A/pod/owned-worker-v825s-off.json" | cut -c1-64)
OW825=("${POD_SSH[@]}" "$POD/runtime-exact/venv/bin/python -I -B $POD/owned-worker-v7.py --config $POD/owned-worker-v825s.json --sha256 $NEW_SHA")
U824="$UNITS/affine-mimo-trainer-supervised.service.d/zzzz~~~~~v824s-owned-worker.conf"
grep -q "owned-worker-v824s.json --sha256 $V824_SHA" "$U824" || die "live trainer drop-in is not v824s"
if [[ ! -e "$A/trainer/$TC" ]]; then
  line=$(grep '^ExecStart="/usr/bin/ssh"' "$U824"); new=${line//owned-worker-v824s.json --sha256 $V824_SHA/owned-worker-v825s.json --sha256 $NEW_SHA}
  [[ "$new" != "$line" ]] || die "ExecStart rewrite"
  printf '%s\n%s\n%s\n%s\n%s\n' "# MiMo v8.2.5: trainer owned worker = v8.2.4 host layer + engine.drop_unresolved=true (unresolved rollouts dropped from loss)." \
    "# Installed by activate_v825.sh 2026-10-21 (authorized by Jacob 11:29 CR). Rollback: see $D/ROLLBACK.md" "[Service]" "ExecStart=" "$new" > "$A/trainer/$TC"
  chmod 600 "$A/trainer/$TC"
fi
log "owned-worker-v825s sha256=$NEW_SHA (off variant $OFF_SHA)"
# pod files (new directory only; nothing existing overwritten)
if ! "${POD_SSH[@]}" "test -e $POD/host-v825"; then
  "${POD_SSH[@]}" "mkdir -m 700 $POD/host-v825"
  (cd "$SG/trainer-host-v825" && tar -cf - .) | "${POD_SSH[@]}" "tar --no-same-owner -C $POD/host-v825 -xf - && chmod -R go-rwx $POD/host-v825"
  "${POD_SSH[@]}" "test ! -e $POD/owned-worker-v825s.json && umask 077 && cat > $POD/owned-worker-v825s.json" < "$A/pod/owned-worker-v825s.json"
  "${POD_SSH[@]}" "test ! -e $POD/owned-worker-v825s-off.json && umask 077 && cat > $POD/owned-worker-v825s-off.json" < "$A/pod/owned-worker-v825s-off.json"
fi
(cd "$SG/trainer-host-v825" && sha256sum $(find . -type f | sort)) > "$A/pod-host-files.sha256"
"${POD_SSH[@]}" "cd $POD/host-v825 && sha256sum -c --quiet" < "$A/pod-host-files.sha256" || die "pod host files digest"
[[ "$("${POD_SSH[@]}" "sha256sum $POD/owned-worker-v825s.json" | cut -c1-64)" == "$NEW_SHA" ]] || die "pod owned-worker-v825s digest"
log "pod files verified; owned-worker --check (launcher admission + original-method pins on the pod source; no model, no claims)"
"${OW825[@]}" --check 2>&1 | tail -3 | tee -a "$D/activate-v825.log" | grep -q '"check": true' || die "owned-worker v825s --check failed (nothing restarted)"
"${POD_SSH[@]}" "$POD/runtime-exact/venv/bin/python -I -B $POD/owned-worker-v7.py --config $POD/owned-worker-v825s-off.json --sha256 $OFF_SHA --check" 2>&1 | tail -3 | tee -a "$D/activate-v825.log" | grep -q '"check": true' || die "off-variant --check failed (nothing restarted)"
[[ "$MODE" == --activate ]] || { log "preflight OK; nothing restarted"; exit 0; }
# backup
TS=$(date -u +%Y%m%dT%H%M%SZ); BK=$D/backup-trainer-$TS; mkdir -m 700 "$BK"
tar -C "$UNITS" -cf "$BK/systemd-user-units.tar" affine-mimo-trainer-supervised.service.d affine-mimo-lifecycle.service.d
cp -a "$STATE/CURRENT.json" "$BK/CURRENT.json.pre"
systemctl --user show affine-mimo-trainer-supervised.service -p ExecStart --value > "$BK/trainer-execstart.pre.txt"
"${OW824[@]}" --status > "$BK/worker-status.pre.json" 2>&1 || true
sha256sum "$BK"/* > "$BK/BACKUP.sha256"; log "backups -> $BK"
inflight && die "job appeared; nothing changed"
install -m 600 "$A/trainer/$TC" "$UNITS/affine-mimo-trainer-supervised.service.d/$TC"
systemctl --user daemon-reload
if inflight; then rm -f "$UNITS/affine-mimo-trainer-supervised.service.d/$TC"; systemctl --user daemon-reload; die "job appeared; drop-in removed, nothing restarted"; fi
systemctl --user stop affine-mimo-trainer-supervised.service
"${OW824[@]}" --stop-owned-after-drain | tee -a "$D/activate-v825.log" || log "note: no live worker"
st=True; for i in $(seq 1 120); do st=$("${OW824[@]}" --status | "$PY" -c "import json,sys;print(json.load(sys.stdin)['alive'])"); [[ "$st" == False ]] && break; sleep 5; done
[[ "$st" == False ]] || die "v824 worker still alive after 10 min (watcher left stopped)"
inflight && log "WARNING: a TRAINING-JOB appeared during drain (it will be claimed by the v825 worker)"
systemctl --user start affine-mimo-trainer-supervised.service
ok=0; for i in $(seq 1 60); do "${OW825[@]}" --status | grep -q '"alive": true' && { ok=1; break; }; sleep 5; done
[[ $ok == 1 ]] || die "v825 worker not alive (run ROLLBACK.md steps)"
"${POD_SSH[@]}" "grep -a 'trainer-host-v825' $POD/owned-worker-v7.log | tail -1 | cut -c1-400" | tee -a "$D/activate-v825.log"
log "ACTIVATED trainer v8.2.5 drop_unresolved=true (cold optimizer restore at the next job)"
