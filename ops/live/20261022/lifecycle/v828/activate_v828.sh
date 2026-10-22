#!/usr/bin/env bash
# MiMo v8.2.8 = signed v8.2.7 composition + audit population off the critical path + audit scheduler disabled + grading 8->32.
# ONLY a new lifecycle drop-in is installed; lifecycle config v8.2.4 (c38a9fcf), policy, closure, trainer, writer, v9 scorer,
# chain UNCHANGED. Nothing existing is overwritten or deleted. Authorized by Jacob 2026-10-21.
#   activate_v828.sh --preflight   verify release + launcher --check; nothing installed
#   activate_v828.sh --arm         1) wait until the v8.2.7 arm logs ACTIVATED (abort if it aborts); 2) wait until that first
#                                  v8.2.7 epoch (e79) FINALIZES with the same lifecycle pid; 3) at the next boundary (new
#                                  epoch OPENING, 0 captures) require e79 clean (finalized+updated, dispatch/job/result/audit
#                                  population present, no main-loop retrying, same pid) AND the operator GO file
#                                  E<n>-VERIFIED.ok; then back up, install drop-in, restart once, verify.
# No chain transactions, no messages, never touches the SUBNETOWNER mnemonic, trainer untouched.
set -euo pipefail
umask 077
D=/home/const/subnet120-rewrite/state/root-audits/20261021-v828-audit-off-path
SG=$D/release-v828-signed; A=$D/activation-v828
D7=/home/const/subnet120-rewrite/state/root-audits/20261021-v827-incremental-capture
STATE=/home/const/subnet120-rewrite/state/mimo-mainnet-v1
UNITS=/home/const/.config/systemd/user; LD=$UNITS/affine-mimo-lifecycle.service.d
SVC=affine-mimo-lifecycle.service
PY=/home/const/subnet120-rewrite/.venv/bin/python
LC="zzzz~~~~~~~~~~~~v828-release.conf"
OP=$SG/v828-operator.ROOT-ADMITTED.private.json
MODE=${1:---preflight}
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$D/activate-v828.log"; }
die(){ log "ABORT: $*"; exit 1; }
cur(){ "$PY" -c "import json;print(json.load(open('$STATE/CURRENT.json'))['$1'])"; }
mainpid(){ systemctl --user show $SVC -p MainPID --value; }
cmdline(){ tr '\0' ' ' < /proc/$(mainpid)/cmdline; }
log "v8.2.8 activation mode=$MODE"
(cd "$SG" && sha256sum --quiet -c SIGNED-MANIFEST.sha256) || die "release-v828-signed manifest mismatch"
grep -q '"checks_passed": true' "$SG/SIGNING-RESULT.json" || die "signing checks did not pass"
mkdir -p -m 700 "$A"
if [[ ! -e "$A/$LC" ]]; then
  printf '%s\n%s\n%s\n%s\n%s\n' \
    "# MiMo v8.2.8: v8.2.7 + audit population off the critical path + audit scheduler disabled + native grading 32. Lifecycle config v8.2.4 unchanged." \
    "# Installed by activate_v828.sh (authorized by Jacob 2026-10-21). Overrides zzzz~~~~~~~~~~~v827-release.conf (left in place). Rollback: $D/ROLLBACK.md" \
    "[Service]" "ExecStart=" \
    "ExecStart=$PY -B $SG/mimo_v828_launcher.py --operator-config $OP" > "$A/$LC"
  chmod 600 "$A/$LC"
fi
test ! -e "$LD/$LC" || die "v8.2.8 drop-in already installed"
"$PY" -B "$SG/mimo_v828_launcher.py" --operator-config "$OP" --check 2>&1 | tail -1 | tee -a "$D/activate-v828.log" | grep -q '"v828": true' || die "launcher --check failed"
systemctl --user is-active --quiet $SVC || die "lifecycle not active"
log "CURRENT epoch_index=$(cur epoch_index) round=$(cur round); lifecycle pid $(mainpid)"
[[ "$MODE" == --preflight ]] && { log "preflight OK; nothing installed"; exit 0; }
[[ "$MODE" == --arm ]] || die "bad mode"
# 1) v8.2.7 must activate first (its own arm, untouched). Abort if it aborts.
base7=$(grep -c . "$D7/activate-v827.log" || true)
log "armed: waiting for v8.2.7 ACTIVATED (activate-v827.log lines now $base7)"
for i in $(seq 1 21600); do
  new=$(tail -n +$((base7+1)) "$D7/activate-v827.log" 2>/dev/null || true)
  grep -q "ACTIVATED lifecycle v8.2.7" <<<"$new" && break
  grep -q "ABORT" <<<"$new" && die "v8.2.7 activation aborted; v8.2.8 not installed"
  grep -q "ACTIVATED lifecycle v8.2.7" "$D7/activate-v827.log" && break
  sleep 2
done
grep -q "ACTIVATED lifecycle v8.2.7" "$D7/activate-v827.log" || die "v8.2.7 not activated within 12 h"
cmdline | grep -q "uid-operator.v8.2.7.ROOT-ADMITTED" || die "lifecycle is not running v8.2.7"
PID7=$(mainpid); EV=$(cur epoch_index); RV=$(cur round)
log "v8.2.7 active: pid $PID7; first v8.2.7 epoch e$EV round $RV; waiting for it to finalize"
SINCE=$(date -u '+%Y-%m-%d %H:%M:%S')
# 2) wait for the first v8.2.7 epoch to finalize, same process throughout
while [[ "$(cur round)" == "$RV" ]]; do
  [[ "$(mainpid)" == "$PID7" ]] || die "lifecycle pid changed during e$EV ($PID7 -> $(mainpid)); not clean"
  sleep 1
done
log "CURRENT advanced: epoch_index=$(cur epoch_index) round=$(cur round)"
EN=$(cur epoch_index); E=$STATE/epochs/$EN; P=$STATE/epochs/$EV
for i in $(seq 1 900); do [[ -e "$E/coordinator/OPENING.json" ]] && break; sleep 1; done
[[ -e "$E/coordinator/OPENING.json" ]] || die "new epoch opening not published within 15 min"
log "e$EN opening published: $(date -u -r $E/coordinator/OPENING.json +%FT%TZ)"
for i in $(seq 1 60); do [[ -e "$E/DISCOVERY.json" ]] && break; sleep 1; done
# 3) e$EV clean on v8.2.7
[[ "$(mainpid)" == "$PID7" ]] || die "lifecycle pid changed ($PID7 -> $(mainpid))"
[[ -e "$P/FINALIZED.json" && -e "$P/TRAINING-JOB.json" && -e "$P/TRAINING-RESULT.json" && -e "$P/AUDIT-POPULATION.json" && -e "$P/coordinator/DISPATCH.json" ]] || die "e$EV artifacts incomplete"
"$PY" - "$P" <<'PYEOF' || die "e$EV finalize/dispatch not clean"
import json,sys
p=sys.argv[1]; f=json.load(open(p+'/FINALIZED.json'))['payload']; d=json.load(open(p+'/coordinator/DISPATCH.json'))['payload']
assert f['result']['status']=='updated', f['result']['status']
assert d['selected_task_count']>0
PYEOF
R=$(journalctl -q --user -u $SVC --since "$SINCE" --no-pager -o cat 2>/dev/null | grep -c '^{"status": "retrying"' || true)
[[ "$R" == 0 ]] || die "e$EV had $R main-loop retrying passes on v8.2.7; not clean"
[[ -e "$D/E$EV-VERIFIED.ok" ]] || die "operator GO file E$EV-VERIFIED.ok absent (manual e$EV verification not recorded); nothing installed"
n=$( (ls "$E/coordinator/captured" 2>/dev/null || true) | wc -l)
[[ $n == 0 ]] || die "e$EN already has $n captures; not restarting"
log "e$EV clean on v8.2.7 (finalized updated, 0 retrying, pid $PID7, GO file present); at boundary e$EN with 0 captures"
TS=$(date -u +%Y%m%dT%H%M%SZ); BK=$D/backup-lifecycle-$TS; mkdir -m 700 "$BK"
tar -C "$UNITS" -cf "$BK/systemd-user-units.tar" affine-mimo-lifecycle.service.d
cp -a "$STATE/CURRENT.json" "$BK/CURRENT.json.pre"
systemctl --user show $SVC -p ExecStart --value > "$BK/lifecycle-execstart.pre.txt"
mainpid > "$BK/lifecycle-mainpid.pre.txt"
sha256sum "$BK"/* > "$BK/BACKUP.sha256"; log "backups -> $BK"
install -m 600 "$A/$LC" "$LD/$LC"
systemctl --user daemon-reload
systemctl --user restart $SVC; log "lifecycle restarted on v8.2.8"
sleep 40
systemctl --user is-active --quiet $SVC || die "lifecycle not active after restart (see ROLLBACK.md)"
pid=$(mainpid); CMD=$(cmdline)
grep -q "v828-operator.ROOT-ADMITTED" <<<"$CMD" && log "lifecycle pid $pid runs v828-operator" || die "lifecycle argv is not v8.2.8"
J=$(journalctl --user -u $SVC --since "-3min" --no-pager 2>/dev/null || true)
grep -q '"component": "host-v824"' <<<"$J" && log "host-v824 overrides installed" || die "no host-v824 install line"
grep -q '"component": "host-v828"' <<<"$J" && log "host-v828 overrides installed" || die "no host-v828 install line"
grep -q '"status": "disabled-v828"' <<<"$J" && log "audit scheduler disabled" || die "no audit-scheduler disabled line"
tail -8 <<<"$J" | cut -c1-300 | tee -a "$D/activate-v828.log"
log "ACTIVATED lifecycle v8.2.8"
