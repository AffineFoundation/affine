#!/usr/bin/env bash
# MiMo v8.2.3 activation (lifecycle-config succession over v8.2.2; only trainer_config.overlong_soft_start 7000->6000).
# Policy dfe0cbf7, chain 460a1f5e, archive a2a03cb3, worker closure d0b0475d, writer, hosts: all UNCHANGED.
# Usage: activate_v823.sh          -> preflight only (read-only)
#        activate_v823.sh --arm    -> while e45 trains (round 234): poll CURRENT every 1 s; at round 235 (e45 finalized)
#             swap the lifecycle drop-in + mining observer drop-in, restart those two units. Aborts (nothing changed) if
#             epochs/46 already has a TRAINING-JOB (the new trainer config must apply from a fresh epoch's job).
# Rollback: rollback_v823.sh. No chain transactions, no Discord, never touches the SUBNETOWNER mnemonic.
set -euo pipefail
umask 077
D=/home/const/subnet120-rewrite/state/root-audits/20261016-v823
SG=$D/release-v823-signed; A=$D/activation-v823
STATE=/home/const/subnet120-rewrite/state/mimo-mainnet-v1
UNITS=/home/const/.config/systemd/user
PY=/home/const/subnet120-rewrite/.venv/bin/python
POL=dfe0cbf78bcf59c4b8ee2162bbd9f573a2e45d22b757c87d820ad7549a4b3eba
PRE_ROUND=234; TARGET_ROUND=235; POST_EI=46
LC="zzzz~~~~~~~~v823-release.conf"; OC="zzzz~~~mimo-observer-v823.conf"
ARM=0; [[ "${1:-}" == "--arm" ]] && ARM=1
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a "${ALOG:-$D/activate-v823.log}"; }
die(){ log "ABORT: $*"; exit 1; }
cur(){ "$PY" -c "import json;print(json.load(open('$STATE/CURRENT.json'))['$1'])"; }
log "v8.2.3 activation (arm=$ARM)"
[[ "$(cur policy_sha256)" == "$POL" ]] || die "live policy is not dfe0cbf7"
(cd "$SG" && sha256sum --quiet -c SIGNED-MANIFEST.sha256) || die "release-v823-signed manifest mismatch"
sha256sum --quiet -c "$A/MANIFEST.sha256" || die "activation-v823 manifest mismatch"
"$PY" -c "import json;r=json.load(open('$SG/SIGNING-RESULT.json'));assert r['checks_passed'] is True and r['policy_sha256']=='$POL'" || die "signing result"
test -f "$UNITS/affine-mimo-lifecycle.service.d/zzzz~~~~~~~v822s-release.conf" || die "expected live v8.2.2 lifecycle drop-in"
test ! -e "$UNITS/affine-mimo-lifecycle.service.d/$LC" || die "v8.2.3 lifecycle drop-in already installed"
test ! -e "$UNITS/affine-mining-discovery.service.d/$OC" || die "v8.2.3 observer drop-in already installed"
systemctl --user is-active --quiet affine-mimo-lifecycle.service || die "lifecycle not active"
systemctl --user is-active --quiet affine-mining-discovery.service || die "mining discovery not active"
ROUND=$(cur round); EI=$(cur epoch_index); log "live CURRENT round=$ROUND epoch_index=$EI"
"$PY" -B /home/const/subnet120-rewrite/state/root-audits/20261013-null-pow-v1/uid1000-host-prospective/mimo_uid_capacity_launcher.py \
  --operator-config "$SG/uid-operator.v8.2.3.ROOT-ADMITTED.private.json" --check | tail -1 | grep -q '"operator_admission_passed": true' || die "launcher --check"
log "preflight OK"
[[ $ARM -eq 1 ]] || { log "preflight only; nothing changed"; exit 0; }
[[ "$ROUND" == "$PRE_ROUND" ]] || die "arm only while round $PRE_ROUND (e45 training); now $ROUND"
log "ARMED: polling for round $TARGET_ROUND"
until [[ "$(cur round 2>/dev/null)" == "$TARGET_ROUND" ]]; do
  r=$(cur round 2>/dev/null || echo "$PRE_ROUND"); (( r > TARGET_ROUND )) && die "round $r already past $TARGET_ROUND"
  sleep 1
done
log "boundary: round $TARGET_ROUND (e45 finalized), epoch_index $(cur epoch_index)"
[[ "$(cur epoch_index)" == "$POST_EI" ]] || die "unexpected epoch_index after boundary; nothing changed"
test ! -e "$STATE/epochs/$POST_EI/TRAINING-JOB.json" || die "epochs/$POST_EI already has a TRAINING-JOB; nothing changed"
TS=$(date -u +%Y%m%dT%H%M%SZ); BK=$D/backup-$TS; mkdir -m 700 "$BK"; ALOG=$BK/ACTIVATION.log; echo "$BK" > "$A/LAST-ACTIVATION"
tar -C "$UNITS" -cf "$BK/systemd-user-units.tar" affine-mimo-lifecycle.service.d affine-mining-discovery.service.d affine-live-reward-writer.service.d
cp -a "$STATE/CURRENT.json" "$BK/CURRENT.json.pre-v823"
systemctl --user show affine-mimo-lifecycle.service -p ExecStart --value > "$BK/lifecycle-execstart.pre-v823.txt"
cp -a /home/const/subnet120/affine/website/mining.json "$BK/mining.json.pre-v823" 2>/dev/null || true
sha256sum "$BK"/*.pre-v823* "$BK/systemd-user-units.tar" > "$BK/BACKUP.sha256"
log "backups -> $BK"
install -m 600 "$A/arbos/affine-mimo-lifecycle.service.d/$LC" "$UNITS/affine-mimo-lifecycle.service.d/$LC"
install -m 600 "$A/arbos/affine-mining-discovery.service.d/$OC" "$UNITS/affine-mining-discovery.service.d/$OC"
systemctl --user daemon-reload
systemctl --user restart affine-mimo-lifecycle.service
log "lifecycle restarted on v8.2.3"
sleep 20
systemctl --user restart affine-mining-discovery.service
log "mining observer restarted on v8.2.3 binding"
sleep 60
systemctl --user is-active --quiet affine-mimo-lifecycle.service || die "lifecycle not active after restart (rollback_v823.sh)"
PID=$(systemctl --user show affine-mimo-lifecycle.service -p MainPID --value)
tr '\0' ' ' < /proc/$PID/cmdline | grep -q "uid-operator.v8.2.3.ROOT-ADMITTED" && log "lifecycle pid $PID runs uid-operator.v8.2.3" || die "lifecycle argv is not v8.2.3"
systemctl --user is-active --quiet affine-mining-discovery.service || log "WARNING: mining observer not active"
journalctl --user -u affine-mimo-lifecycle.service --since "-2min" --no-pager | tail -15 | cut -c1-300 | tee -a "$ALOG"
journalctl --user -u affine-mining-discovery.service --since "-1min" --no-pager | tail -5 | cut -c1-300 | tee -a "$ALOG"
"$PY" -c "import json;d=json.load(open('/home/const/subnet120/affine/website/mining.json'));print('mining.json', {k:d.get(k) for k in ('status','epoch','error','accepting_submissions')})" 2>&1 | tee -a "$ALOG" || true
log "ACTIVATED v8.2.3 at round $TARGET_ROUND (e$POST_EI is the first epoch trained with overlong_soft_start 6000). Rollback: $D/rollback_v823.sh"
