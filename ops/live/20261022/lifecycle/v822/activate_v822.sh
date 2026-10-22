#!/usr/bin/env bash
# MiMo v8.2.2 successor activation (chain v7 -> v8.2 -> v8.2.1 -> v8.2.2), SN120 Arbos.  PREPARED, NOT RUN.
# v8.2.2 = v8.2.1 + archive-scoped audit selection (ops/mimo_lifecycle.py a7bc7f77; worker closure unchanged d0b0475d),
#          audit retries 3, max_tasks per SUCCESSOR-BINDING (128), job_seconds 7200.
# Usage:  activate_v822.sh                -> preflight only (read-only + full dry-run on fresh COPIES; writes PREFLIGHT-PASS)
#         activate_v822.sh --arm [--allow-move-unopened-plan]
#              start while epoch PRE_EI trains (CURRENT.round == TARGET_ROUND-1). TARGET_ROUND = signed first_round.
#              Polls CURRENT.json every 1 s; at the boundary it stops the lifecycle at once. If the next epoch already has an
#              OPENING it restarts the v8.2.1 lifecycle and aborts (nothing changed). An unopened PLAN is moved into the backup
#              ONLY with --allow-move-unopened-plan; otherwise restart v8.2.1 and abort (nothing changed).
# Rollback: rollback_v822.sh (backup dir recorded in activation-v822/LAST-ACTIVATION).
# No chain transactions, no Discord, never touches the SUBNETOWNER mnemonic.
set -euo pipefail
umask 077
D=/home/const/subnet120-rewrite/state/root-audits/20261015-mimo-v8.2-release-succession
SG=$D/release-v822-signed; A=$D/activation-v822
STATE=/home/const/subnet120-rewrite/state/mimo-mainnet-v1
UNITS=/home/const/.config/systemd/user
PY=/home/const/subnet120-rewrite/.venv/bin/python
ROOT=3301134b38401196d006a621ac4a772bb4b0e6afa15a7d34a0d1ae5f2c630bcd
TARGET_ROUND=$("$PY" -c "import json;print(json.load(open('$SG/SIGNING-RESULT.json'))['first_round'])")
NEWPOL=$("$PY" -c "import json;print(json.load(open('$SG/SIGNING-RESULT.json'))['policy_sha256'])")
MAXT=$("$PY" -c "import json;print(json.load(open('$D/staged-v822-successor/SUCCESSOR-BINDING.json'))['max_tasks'])")
OLDPOL=4f0171068b024b876992698fd32e95aed0d19486c98b3721ab5803a51adb3476
PRE_EI=$((TARGET_ROUND-1-189)); POST_EI=$((TARGET_ROUND-189))
ROLES=(trainer evaluator verifier-1 verifier-2 verifier-3)
TRAINER_BASE=/workspace/affine-mimo-trainer-prod
TRAINER_STATE=$TRAINER_BASE/trainer-state/affine-mimo9b-grpo-20261013
ARM=0; ALLOW_MOVE=0
for x in "$@"; do case "$x" in --arm) ARM=1;; --allow-move-unopened-plan) ALLOW_MOVE=1;; *) echo "unknown arg $x"; exit 2;; esac; done
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a "${ALOG:-/dev/null}"; }
die(){ log "ABORT: $*"; exit 1; }
plan(){ "$PY" -c "import json,sys;print(json.load(open('$A/ACTIVATION-PLAN.json'))['hosts'][sys.argv[1]][sys.argv[2]])" "$1" "$2"; }
rssh(){ local r=$1; shift; ssh -T -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10 \
  -o UserKnownHostsFile=/home/const/.lium/known_hosts/$(plan "$r" known_hosts) -p "$(plan "$r" port)" root@"$(plan "$r" host)" "$@"; }
rscp(){ local r=$1; scp -q -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10 \
  -o UserKnownHostsFile=/home/const/.lium/known_hosts/$(plan "$r" known_hosts) -P "$(plan "$r" port)" "$2" root@"$(plan "$r" host)":"$3"; }
cur(){ "$PY" -c "import json;c=json.load(open('$STATE/CURRENT.json'));print(c['$1'])"; }
# ---------------- PREFLIGHT (read-only) ----------------
log "v8.2.2 activation preflight (arm=$ARM, allow_move=$ALLOW_MOVE, target round $TARGET_ROUND)"
[[ "$(cur policy_sha256)" == "$OLDPOL" ]] || die "live policy is not v8.2.1 (4f017106)"
sha256sum --quiet -c "$A/MANIFEST.sha256" || die "activation-v822 manifest mismatch"
(cd "$SG" && sha256sum --quiet -c SIGNED-MANIFEST.sha256) || die "release-v822-signed manifest mismatch"
"$PY" - "$SG" "$D" <<'PYEOF' || die "signing result not clean"
import json,sys;r=json.load(open(sys.argv[1]+'/SIGNING-RESULT.json'));b=json.load(open(sys.argv[2]+'/staged-v822-successor/SUCCESSOR-BINDING.json'))
assert r['dry_run'] is False and r['checks_passed'] is True and r['first_round']>229 and r['authority'].startswith('3301134b') and r['archive_sha256']==b['archive_sha256'],r
assert b['max_execution_retries']==3 and b['job_seconds']==7200 and b['changed_files']==['ops/mimo_lifecycle.py'] and b['source_inventory_sha256'].startswith('d0b0475d'),b
PYEOF
for r in "${ROLES[@]}"; do
  base=$(plan "$r" base)
  rssh "$r" "test -d $(plan "$r" source_v822) && test -f $base/source-registry.PREPARED-v822s.json && test -f $(plan "$r" config_now)" || die "$r: v8.2.2 source/registry not staged"
  test ! -e "$UNITS/affine-mimo-$r-supervised.service.d/zzzz~~~~v822s-owned-worker.conf" || die "$r: v8.2.2 drop-in already installed"
done
test -f "$UNITS/affine-mimo-lifecycle.service.d/zzzz~~~~~~v821s-release.conf" || die "expected live v8.2.1 lifecycle drop-in"
test ! -e "$UNITS/affine-mimo-lifecycle.service.d/zzzz~~~~~~~v822s-release.conf" || die "v8.2.2 lifecycle drop-in already installed"
rssh trainer "test -f $TRAINER_BASE/private/trainer-role.json && test -f $TRAINER_STATE/CURRENT.json" || die "trainer role/state missing"
systemctl --user is-active --quiet affine-mimo-lifecycle.service || die "lifecycle not active before activation"
for r in "${ROLES[@]}"; do
  # worker watchers may flap (verifiers crash on renewed v7 audit jobs in source_guard; this release stops renewing them)
  st=$(systemctl --user show affine-mimo-$r-supervised.service -p ActiveState --value)
  [[ "$st" == active || "$st" == activating ]] || die "affine-mimo-$r-supervised is $st before activation"
  [[ "$st" == active ]] || log "note: affine-mimo-$r-supervised is $st (restart loop); resolved by this activation"
done
ROUND=$(cur round); EI=$(cur epoch_index); log "live CURRENT round=$ROUND epoch_index=$EI"
if [[ $ARM -eq 1 ]]; then
  [[ -f "$A/PREFLIGHT-PASS" ]] && (( $(date +%s) - $(stat -c %Y "$A/PREFLIGHT-PASS") < 10800 )) || die "no preflight PASS in the last 3 h; run activate_v822.sh (no args) first"
  log "using preflight PASS: $(cat "$A/PREFLIGHT-PASS")"
else
  DR=$(mktemp -d "$D/v822-preflight-XXXX"); mkdir -p "$DR/state/auditor" "$DR/out"
  cp -a "$STATE/CURRENT.json" "$DR/state/"; cp -a "$STATE/evaluator" "$DR/state/"
  find "$STATE/auditor" -maxdepth 1 -name '*.population.json' -exec cp -a -t "$DR/state/auditor/" {} +
  cp -a "$STATE/auditor/state.json" "$DR/state/auditor/state.json"
  rssh trainer "cat $TRAINER_STATE/CURRENT.json" > "$DR/trainer-CURRENT.copy.json"
  cp -a "$STATE/epochs/39/coordinator/OPENING.json" "$DR/state/e39-OPENING.json"
  log "preflight dry-run -> $DR/out"
  "$PY" -B "$D/dry_run_v822.py" --signed "$SG" --copy "$DR/state" --trainer-current "$DR/trainer-CURRENT.copy.json" --out "$DR/out" >"$DR/dry-run.log" 2>&1 \
    || die "preflight dry-run failed (see $DR/dry-run.log)"
  "$PY" -c "import json;assert json.load(open('$DR/out/DRY-RUN-RESULT-v822.json'))['passed'] is True" || die "preflight dry-run not passed"
  log "preflight dry-run PASS"; echo "$(date -u +%FT%TZ) $DR/out/DRY-RUN-RESULT-v822.json" > "$A/PREFLIGHT-PASS"
  log "preflight complete; nothing changed. Arm with: activate_v822.sh --arm [--allow-move-unopened-plan]  (while round $((TARGET_ROUND-1)), within 3 h)"; exit 0
fi
[[ "$ROUND" == "$((TARGET_ROUND-1))" && "$EI" == "$PRE_EI" ]] || die "arm only while e$PRE_EI trains (round $((TARGET_ROUND-1))); now round $ROUND epoch $EI"
log "ARMED: polling CURRENT.json every 1 s for round $TARGET_ROUND"
until [[ "$(cur round 2>/dev/null)" == "$TARGET_ROUND" ]]; do sleep 1; done
systemctl --user stop affine-mimo-lifecycle.service
log "boundary seen (e$PRE_EI finalized); v8.2.1 lifecycle stopped"
EI=$(cur epoch_index); [[ "$EI" == "$POST_EI" ]] || die "unexpected epoch_index $EI after boundary (lifecycle left stopped)"
if [[ -e "$STATE/epochs/$EI/coordinator/OPENING.json" ]]; then
  systemctl --user start affine-mimo-lifecycle.service
  die "v8.2.1 already opened epoch $EI; v8.2.1 lifecycle restarted; NOTHING changed. Re-sign for a later round."
fi
if [[ -d "$STATE/epochs/$EI" && $ALLOW_MOVE -ne 1 ]]; then
  systemctl --user start affine-mimo-lifecycle.service
  die "epochs/$EI already planned under v8.2.1 and --allow-move-unopened-plan not given; v8.2.1 lifecycle restarted; NOTHING changed."
fi
# ---------------- BACKUPS ----------------
TS=$(date -u +%Y%m%dT%H%M%SZ); BK=$D/activation-v822-backup-$TS; mkdir -m 700 "$BK"; ALOG=$BK/ACTIVATION.log
echo "$BK" > "$A/LAST-ACTIVATION"; log "backups -> $BK"
tar -C "$UNITS" -cf "$BK/systemd-user-units.tar" affine-mimo-lifecycle.service.d affine-live-reward-writer.service.d \
  $(for r in "${ROLES[@]}"; do echo affine-mimo-$r-supervised.service.d; done)
cp -a "$STATE/CURRENT.json" "$BK/CURRENT.json.pre-v822"
if [[ -d "$STATE/epochs/$EI" ]]; then
  "$PY" -c "import json;p=json.load(open('$STATE/epochs/$EI/PLAN.json'));c=json.load(open('$STATE/CURRENT.json'));assert p['input']==c,'in-flight plan is not for this cursor'" || die "epochs/$EI PLAN does not match CURRENT; investigate (lifecycle left stopped)"
  mv "$STATE/epochs/$EI" "$BK/epoch-$EI.v821-unopened"; echo "EPOCH_MOVED=$EI" >> "$BK/ACTIVATION-RECORD.env.pending"
  log "moved unopened v8.2.1 epochs/$EI -> $BK/epoch-$EI.v821-unopened (--allow-move-unopened-plan)"
fi
[[ -f "$STATE/evaluator/state.json" ]] && cp -a "$STATE/evaluator/state.json" "$BK/evaluator-state.json.pre-v822"
cp -a "$STATE/roles.sqlite3" "$BK/roles.sqlite3.pre-v822"
[[ -e "$STATE/roles.sqlite3-journal" ]] && cp -a "$STATE/roles.sqlite3-journal" "$BK/roles.sqlite3-journal.pre-v822" || true
rssh trainer "cp -a $TRAINER_STATE/CURRENT.json $TRAINER_STATE/CURRENT.json.bak-v822s-$TS && cp -a $TRAINER_BASE/private/trainer-role.json $TRAINER_BASE/private/trainer-role.json.bak-v822s-$TS"
rssh trainer "cat $TRAINER_STATE/CURRENT.json" > "$BK/trainer-CURRENT.json.pre-v822"
{ echo "TS=$TS"; cat "$BK/ACTIVATION-RECORD.env.pending" 2>/dev/null || true; } > "$BK/ACTIVATION-RECORD.env"; rm -f "$BK/ACTIVATION-RECORD.env.pending"
sha256sum "$BK"/*.pre-v822 > "$BK/BACKUP.sha256"
# ---------------- R2 SOURCE ARCHIVE (write-once, read-back) ----------------
"$PY" -B "$D/upload_v822_source.py" | tee -a "$ALOG" | grep -q '"readback_verified": true' || die "R2 source upload/readback failed (lifecycle stopped; rollback_v822.sh)"
# ---------------- STAGE HOST FILES ----------------
for r in "${ROLES[@]}"; do
  base=$(plan "$r" base)
  rssh "$r" "umask 077; cat > $base/owned-worker-v822s.json" < "$A/hosts/$r/owned-worker-v822s.json"
  rssh "$r" "umask 077; cat > $base/source-registry.qualified-v822s.json" < "$A/hosts/$r/source-registry.qualified-v822s.json"
  rssh "$r" "echo '$(plan "$r" config_v822s_sha256)  $base/owned-worker-v822s.json' | sha256sum -c --quiet && echo '$(plan "$r" registry_v822s_sha256)  $base/source-registry.qualified-v822s.json' | sha256sum -c --quiet" || die "$r: staged file hash"
  rssh "$r" "$(plan "$r" owpy) --config $base/owned-worker-v822s.json --sha256 $(plan "$r" config_v822s_sha256) --check" >/dev/null || die "$r: v8.2.2 worker --check failed"
  log "$r: owned-worker-v822s.json + registry staged, --check OK"
done
rscp trainer "$D/merge_trainer_role_v821.py" /root/merge_trainer_role_v821.py
rssh trainer "python3 /root/merge_trainer_role_v821.py && rm -f /root/merge_trainer_role_v821.py" < "$SG/TRAINER-ROLE-RELEASE-SUCCESSION.v8.2.2.private.json" | tee -a "$ALOG"
# ---------------- DRAIN CURRENT WORKERS ----------------
for r in "${ROLES[@]}"; do systemctl --user stop affine-mimo-$r-supervised.service; done
for r in "${ROLES[@]}"; do
  rssh "$r" "$(plan "$r" owpy) --config $(plan "$r" config_now) --sha256 $(plan "$r" config_now_sha256) --stop-owned-after-drain" >>"$ALOG" 2>&1 || log "$r: no live worker to stop"
done
for r in "${ROLES[@]}"; do
  st=True
  for i in $(seq 1 180); do
    st=$(rssh "$r" "$(plan "$r" owpy) --config $(plan "$r" config_now) --sha256 $(plan "$r" config_now_sha256) --status" | "$PY" -c "import json,sys;print(json.load(sys.stdin)['alive'])")
    [[ "$st" == "False" ]] && break; sleep 5
  done
  [[ "$st" == "False" ]] || die "$r: worker still alive after 15 min (rollback_v822.sh to revert)"
  log "$r: worker drained"
done
# ---------------- INSTALL DROP-INS + START ----------------
for r in "${ROLES[@]}"; do
  install -D -m 600 "$A/hosts/$r/zzzz~~~~v822s-owned-worker.conf" "$UNITS/affine-mimo-$r-supervised.service.d/zzzz~~~~v822s-owned-worker.conf"
done
install -m 600 "$A/arbos/affine-mimo-lifecycle.service.d/zzzz~~~~~~~v822s-release.conf" "$UNITS/affine-mimo-lifecycle.service.d/"
install -m 600 "$A/arbos/affine-live-reward-writer.service.d/zzzz~~~~~mimo-v822-population-index.conf" "$UNITS/affine-live-reward-writer.service.d/"
systemctl --user daemon-reload
for r in "${ROLES[@]}"; do systemctl --user start affine-mimo-$r-supervised.service; done
systemctl --user start affine-mimo-lifecycle.service
log "units started; verifying"
sleep 75
for r in "${ROLES[@]}"; do
  # A watcher may restart a few times while a slow worker finishes startup ("startup incomplete" lock); the evaluator took
  # ~95 s on 2026-10-16. Accept active|activating while the v8.2.2 worker reports alive; give up after 5 min.
  okr=0; st=unknown
  for i in $(seq 1 30); do
    st=$(systemctl --user show affine-mimo-$r-supervised.service -p ActiveState --value)
    if [[ "$st" == active || "$st" == activating ]] && rssh "$r" "$(plan "$r" owpy) --config $(plan "$r" config_v822s) --sha256 $(plan "$r" config_v822s_sha256) --status" | tee -a "$ALOG" | grep -q '"alive": true'; then okr=1; break; fi
    sleep 10
  done
  [[ $okr == 1 ]] || die "$r: v8.2.2 worker not alive (unit $st) after 5 min"
  [[ "$st" == active ]] && log "$r: unit active, v8.2.2 worker alive" || log "note: $r watcher $st (re-attaching); v8.2.2 worker alive"
done
systemctl --user is-active --quiet affine-mimo-lifecycle.service || die "lifecycle not active"
journalctl --user -u affine-mimo-lifecycle.service --since "-3min" --no-pager | tail -20 | cut -c1-300 | tee -a "$ALOG"
"$PY" -c "import json;c=json.load(open('$STATE/CURRENT.json'));assert c['policy_sha256']=='$NEWPOL' and c['max_tasks']==$MAXT,(c['policy_sha256'][:12],c['max_tasks'])" \
  && log "CURRENT adopted v8.2.2 (max_tasks $MAXT)" || log "WARNING: CURRENT not yet adopted (check lifecycle journal)"
"$PY" -B -I /home/const/subnet120-rewrite/state/root-audits/20261014-community-weights-check-v1/parallel-evidence-v6/hourly_population_index_entry_v6.py \
  --policy "$SG/HOST-COMPOSITION.v8.2.2-accel.ROOT-SIGNED.private.json" --authority $ROOT --check | tail -1 | tee -a "$ALOG"
log "ACTIVATED v8.2.2 (first_round $TARGET_ROUND). Backups: $BK. Rollback: $D/rollback_v822.sh"
