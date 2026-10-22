#!/usr/bin/env bash
# MiMo v8.2.6 lifecycle host capture layer (deadline re-check, CapturePass v826, workers 16). ONLY the lifecycle drop-in changes;
# lifecycle config v8.2.4 (c38a9fcf), policy, closure, trainer, writer, v9 scorer, chain UNCHANGED. Nothing existing is overwritten.
#   activate_v826.sh --preflight   verify release + launcher --check; nothing installed
#   activate_v826.sh --arm         wait (1 s poll) until CURRENT round advances past ROUND0 (current epoch finalized), then
#                                  back up, install drop-in, restart lifecycle once (between epochs), verify
# No chain transactions, no messages, never touches the SUBNETOWNER mnemonic, trainer untouched.
set -euo pipefail
umask 077
D=/home/const/subnet120-rewrite/state/root-audits/20261021-v826-capture-deadline
SG=$D/release-v826-signed; A=$D/activation-v826
STATE=/home/const/subnet120-rewrite/state/mimo-mainnet-v1
UNITS=/home/const/.config/systemd/user; LD=$UNITS/affine-mimo-lifecycle.service.d
PY=/home/const/subnet120-rewrite/.venv/bin/python
LC="zzzz~~~~~~~~~~v826-release.conf"
UIDOP=$SG/uid-operator.v8.2.6.ROOT-ADMITTED.private.json
MODE=${1:---preflight}
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$D/activate-v826.log"; }
die(){ log "ABORT: $*"; exit 1; }
cur(){ "$PY" -c "import json;print(json.load(open('$STATE/CURRENT.json'))['$1'])"; }
log "v8.2.6 activation mode=$MODE"
(cd "$SG" && sha256sum --quiet -c SIGNED-MANIFEST.sha256) || die "release-v826-signed manifest mismatch"
grep -q '"checks_passed": true' "$SG/SIGNING-RESULT.json" || die "signing checks did not pass"
mkdir -p -m 700 "$A"
if [[ ! -e "$A/$LC" ]]; then
  printf '%s\n%s\n%s\n%s\n%s\n' \
    "# MiMo v8.2.6: v8.2.4 composition + host capture v2 (post-pass deadline re-check, CapturePass v826, workers 16). Lifecycle config v8.2.4 unchanged." \
    "# Installed by activate_v826.sh (authorized by Jacob 2026-10-21 12:08 CR). Overrides zzzz~~~~~~~~~v824-release.conf (left in place). Rollback: $D/ROLLBACK.md" \
    "[Service]" "ExecStart=" \
    "ExecStart=$PY -B $SG/mimo_uid_capacity_launcher.py --operator-config $UIDOP" > "$A/$LC"
  chmod 600 "$A/$LC"
fi
test ! -e "$LD/$LC" || die "v8.2.6 drop-in already installed"
"$PY" -B "$SG/mimo_uid_capacity_launcher.py" --operator-config "$UIDOP" --check 2>&1 | tail -1 | tee -a "$D/activate-v826.log" | grep -q '"operator_admission_passed": true' || die "launcher --check failed"
systemctl --user is-active --quiet affine-mimo-lifecycle.service || die "lifecycle not active"
ROUND0=$(cur round); EI0=$(cur epoch_index); log "CURRENT epoch_index=$EI0 round=$ROUND0"
[[ "$MODE" == --preflight ]] && { log "preflight OK; nothing installed"; exit 0; }
if [[ "$MODE" == --arm ]]; then
  log "armed: waiting for e$EI0 to finalize (round > $ROUND0)"
  while [[ "$(cur round)" == "$ROUND0" ]]; do sleep 1; done
  log "CURRENT advanced: epoch_index=$(cur epoch_index) round=$(cur round)"
elif [[ "$MODE" == --boundary ]]; then
  # previous epoch finalized; current epoch not yet planned, or opened < 10 min ago with nothing captured yet
  E=$STATE/epochs/$EI0; [[ -e "$STATE/epochs/$((EI0-1))/FINALIZED.json" ]] || die "previous epoch not finalized"
  if [[ -e "$E/PLAN.json" ]]; then
    st=$("$PY" -c "import json;print(json.load(open('$E/PLAN.json'))['manifest']['start'])")
    (( $(date +%s) - st < 600 )) || die "e$EI0 opened $(( $(date +%s) - st )) s ago; not at the boundary (target the next one with --arm)"
    n=$(ls "$E/coordinator/captured" 2>/dev/null | wc -l); log "e$EI0 captures so far: $n (restart is idempotent pre-deadline)"
    [[ ! -e "$E/coordinator/POPULATION.json" ]] || die "sealed"
  fi
  log "at boundary: e$EI0 plan=$( [[ -e $E/PLAN.json ]] && echo yes || echo no )"
else die "bad mode"; fi
TS=$(date -u +%Y%m%dT%H%M%SZ); BK=$D/backup-lifecycle-$TS; mkdir -m 700 "$BK"
tar -C "$UNITS" -cf "$BK/systemd-user-units.tar" affine-mimo-lifecycle.service.d
cp -a "$STATE/CURRENT.json" "$BK/CURRENT.json.pre"
systemctl --user show affine-mimo-lifecycle.service -p ExecStart --value > "$BK/lifecycle-execstart.pre.txt"
systemctl --user show affine-mimo-lifecycle.service -p MainPID --value > "$BK/lifecycle-mainpid.pre.txt"
sha256sum "$BK"/* > "$BK/BACKUP.sha256"; log "backups -> $BK"
install -m 600 "$A/$LC" "$LD/$LC"
systemctl --user daemon-reload
systemctl --user restart affine-mimo-lifecycle.service; log "lifecycle restarted on v8.2.6"
sleep 30
systemctl --user is-active --quiet affine-mimo-lifecycle.service || die "lifecycle not active after restart (see ROLLBACK.md)"
pid=$(systemctl --user show affine-mimo-lifecycle.service -p MainPID --value)
tr '\0' ' ' < /proc/$pid/cmdline | grep -q "uid-operator.v8.2.6.ROOT-ADMITTED" && log "lifecycle pid $pid runs uid-operator.v8.2.6" || die "lifecycle argv is not v8.2.6"
journalctl --user -u affine-mimo-lifecycle.service --since "-2min" --no-pager | grep -q '"component": "host-v824"' && log "host-v824 overrides installed" || die "no host-v824 install line"
journalctl --user -u affine-mimo-lifecycle.service --since "-1min" --no-pager | tail -8 | cut -c1-300 | tee -a "$D/activate-v826.log"
log "ACTIVATED lifecycle v8.2.6"
