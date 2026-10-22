#!/usr/bin/env bash
# MiMo v8.2.4 activation (ROOT-admitted host layers; policy dfe0cbf7 / closure d0b0475d / archive / writer UNCHANGED).
# Usage:
#   activate_v824.sh                      preflight only (read-only)
#   activate_v824.sh --arm-lifecycle      while e51 trains (round 240): poll CURRENT every 1 s; at round 241 (e51 finalized)
#                                         install lifecycle + observer drop-ins of variant v8.2.4-t128, restart both units.
#   activate_v824.sh --lifecycle TAG      install variant TAG (v8.2.4-t128 | v8.2.4) now (used for 256 while an epoch trains;
#                                         the in-flight epoch keeps its planned max_tasks; the next boundary re-stamps).
#   activate_v824.sh --trainer            pod host files + owned-worker-v824s.json, owned-worker --check, then (only if the
#                                         trainer is idle: no TRAINING-JOB without TRAINING-RESULT) drain-restart the worker.
# Rollback: rollback_v824.sh. No chain transactions, no Discord, never touches the SUBNETOWNER mnemonic.
set -euo pipefail
umask 077
D=/home/const/subnet120-rewrite/state/root-audits/20261017-v824
SG=$D/release-v824-signed; A=$D/activation-v824
STATE=/home/const/subnet120-rewrite/state/mimo-mainnet-v1
UNITS=/home/const/.config/systemd/user
PY=/home/const/subnet120-rewrite/.venv/bin/python
POL=dfe0cbf78bcf59c4b8ee2162bbd9f573a2e45d22b757c87d820ad7549a4b3eba
PRE_ROUND=240; TARGET_ROUND=241; POST_EI=52
LC="zzzz~~~~~~~~~v824-release.conf"; OC="zzzz~~~~mimo-observer-v824.conf"; TC="zzzz~~~~~v824s-owned-worker.conf"
POD_SSH=(ssh -T -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10 -o UserKnownHostsFile=/home/const/.lium/known_hosts/0a7aba3d-9443-445b-9601-c78ac0d14b1c -p 40301 root@89.169.120.92)
POD=/workspace/affine-mimo-trainer-prod
MODE=preflight; TAG=""
case "${1:-}" in --arm-lifecycle) MODE=arm; TAG=v8.2.4-t128;; --lifecycle) MODE=now; TAG="${2:?tag}";; --trainer) MODE=trainer;; --trainer-restart) MODE=trestart;; "") ;; *) echo "bad args"; exit 2;; esac
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a "${ALOG:-$D/activate-v824.log}"; }
die(){ log "ABORT: $*"; exit 1; }
cur(){ "$PY" -c "import json;print(json.load(open('$STATE/CURRENT.json'))['$1'])"; }
log "v8.2.4 activation mode=$MODE tag=$TAG"
[[ "$(cur policy_sha256)" == "$POL" ]] || die "live policy is not dfe0cbf7"
(cd "$SG" && sha256sum --quiet -c SIGNED-MANIFEST.sha256) || die "release-v824-signed manifest mismatch"
sha256sum --quiet -c "$A/MANIFEST.sha256" || die "activation-v824 manifest mismatch"
"$PY" -c "import json;r=json.load(open('$SG/SIGNING-RESULT.json'));assert r['checks_passed'] is True and r['policy_sha256']=='$POL'" || die "signing result"
systemctl --user is-active --quiet affine-mimo-lifecycle.service || die "lifecycle not active"
systemctl --user is-active --quiet affine-mining-discovery.service || die "mining discovery not active"
ROUND=$(cur round); EI=$(cur epoch_index); log "live CURRENT round=$ROUND epoch_index=$EI max_tasks=$(cur max_tasks)"
for t in v8.2.4-t128 v8.2.4; do
  "$PY" -B "$SG/mimo_uid_capacity_launcher.py" --operator-config "$SG/uid-operator.$t.ROOT-ADMITTED.private.json" --check | tail -1 \
    | grep -q '"v824_override_pins_passed": true' || die "launcher --check $t"
done
log "preflight OK"
backup(){ TS=$(date -u +%Y%m%dT%H%M%SZ); BK=$D/backup-$1-$TS; mkdir -m 700 "$BK"; ALOG=$BK/ACTIVATION.log; echo "$BK" > "$A/LAST-ACTIVATION"
  tar -C "$UNITS" -cf "$BK/systemd-user-units.tar" affine-mimo-lifecycle.service.d affine-mining-discovery.service.d affine-mimo-trainer-supervised.service.d
  cp -a "$STATE/CURRENT.json" "$BK/CURRENT.json.pre"; systemctl --user show affine-mimo-lifecycle.service -p ExecStart --value > "$BK/lifecycle-execstart.pre.txt"
  sha256sum "$BK"/* > "$BK/BACKUP.sha256"; log "backups -> $BK"; }
swap_lifecycle(){ local tag=$1
  install -m 600 "$A/$tag/affine-mimo-lifecycle.service.d/$LC" "$UNITS/affine-mimo-lifecycle.service.d/$LC"
  install -m 600 "$A/$tag/affine-mining-discovery.service.d/$OC" "$UNITS/affine-mining-discovery.service.d/$OC"
  systemctl --user daemon-reload
  systemctl --user restart affine-mimo-lifecycle.service; log "lifecycle restarted on $tag"
  sleep 20; systemctl --user restart affine-mining-discovery.service; log "mining observer restarted on $tag binding"
  sleep 60
  systemctl --user is-active --quiet affine-mimo-lifecycle.service || die "lifecycle not active after restart (run rollback_v824.sh)"
  local pid; pid=$(systemctl --user show affine-mimo-lifecycle.service -p MainPID --value)
  tr '\0' ' ' < /proc/$pid/cmdline | grep -q "uid-operator.$tag.ROOT-ADMITTED" && log "lifecycle pid $pid runs uid-operator.$tag" || die "lifecycle argv is not $tag"
  journalctl --user -u affine-mimo-lifecycle.service --since "-2min" --no-pager | grep -q '"component": "host-v824"' && log "host-v824 overrides installed" || die "no host-v824 install line"
  systemctl --user is-active --quiet affine-mining-discovery.service || log "WARNING: mining observer not active"
  journalctl --user -u affine-mimo-lifecycle.service --since "-2min" --no-pager | tail -12 | cut -c1-300 | tee -a "$ALOG"
  journalctl --user -u affine-mining-discovery.service --since "-1min" --no-pager | tail -4 | cut -c1-300 | tee -a "$ALOG"
  "$PY" -c "import json;d=json.load(open('/home/const/subnet120/affine/website/mining.json'));print('mining.json', {k:d.get(k) for k in ('status','epoch','error','accepting_submissions')})" 2>&1 | tee -a "$ALOG" || true
}
case $MODE in
preflight) log "preflight only; nothing changed"; exit 0;;
arm)
  [[ "$ROUND" == "$PRE_ROUND" ]] || die "arm only while round $PRE_ROUND (e51 training); now $ROUND"
  test ! -e "$UNITS/affine-mimo-lifecycle.service.d/$LC" || die "v8.2.4 lifecycle drop-in already installed"
  log "ARMED: polling for round $TARGET_ROUND"
  until [[ "$(cur round 2>/dev/null)" == "$TARGET_ROUND" ]]; do
    r=$(cur round 2>/dev/null || echo "$PRE_ROUND"); (( r > TARGET_ROUND )) && die "round $r already past $TARGET_ROUND"
    sleep 1
  done
  log "boundary: round $TARGET_ROUND (e51 finalized), epoch_index $(cur epoch_index)"
  [[ "$(cur epoch_index)" == "$POST_EI" ]] || die "unexpected epoch_index after boundary; nothing changed"
  test ! -e "$STATE/epochs/$POST_EI/TRAINING-JOB.json" || die "epochs/$POST_EI already has a TRAINING-JOB; nothing changed"
  backup lifecycle; swap_lifecycle "$TAG"
  log "ACTIVATED lifecycle $TAG at round $TARGET_ROUND. Next: activate_v824.sh --trainer (before the e$POST_EI TRAINING-JOB)";;
now)
  [[ "$TAG" == v8.2.4-t128 || "$TAG" == v8.2.4 ]] || die "unknown tag"
  backup "lifecycle-$TAG"; swap_lifecycle "$TAG"; log "ACTIVATED lifecycle $TAG (round $ROUND)"
  # Lesson e46 (10-16) + e54 (10-17): a warm trainer keeps CPUAdamW bound to the OLD resolved TrainerConfig
  # (max_tasks, soft_start, ...); train_update raises ValueError(optimizer.config != config) and leaves an
  # updating journal => RecoveryRequired. EVERY lifecycle trainer_config change needs a trainer drain-restart
  # after the in-flight job (if any) finishes and BEFORE the next TRAINING-JOB.
  date -u +%FT%TZ > "$A/TRAINER-RESTART-REQUIRED"
  log "REQUIRED NEXT: ./activate_v824.sh --trainer-restart once no TRAINING-JOB is in flight and before the next one (marker $A/TRAINER-RESTART-REQUIRED)"
  if ! [[ -e "$STATE/epochs/$EI/TRAINING-JOB.json" && ! -e "$STATE/epochs/$EI/TRAINING-RESULT.json" ]]; then
    "$0" --trainer-restart || die "trainer-restart failed after lifecycle swap (lifecycle is on $TAG)"
  fi;;
trestart)
  NEW_SHA=$("$PY" -c "import json;print(json.load(open('$A/PREPARE-RESULT.json'))['owned_worker_v824s_sha256'])")
  OW=("${POD_SSH[@]}" "$POD/runtime-exact/venv/bin/python -I -B $POD/owned-worker-v7.py --config $POD/owned-worker-v824s.json --sha256 $NEW_SHA")
  EI=$(cur epoch_index)
  [[ -e "$STATE/epochs/$EI/TRAINING-JOB.json" && ! -e "$STATE/epochs/$EI/TRAINING-RESULT.json" ]] && die "e$EI TRAINING-JOB in flight; never restart the trainer mid-job"
  # stop the watcher FIRST so systemd cannot race the old owner lock (10-17 14:54 CR lesson)
  systemctl --user stop affine-mimo-trainer-supervised.service
  "${OW[@]}" --stop-owned-after-drain | tee -a "${ALOG:-$D/activate-v824.log}" || log "note: no live worker"
  st=True; for i in $(seq 1 120); do st=$("${OW[@]}" --status | "$PY" -c "import json,sys;print(json.load(sys.stdin)['alive'])"); [[ "$st" == False ]] && break; sleep 5; done
  [[ "$st" == False ]] || die "trainer worker still alive after 10 min (watcher left stopped)"
  [[ -e "$STATE/epochs/$EI/TRAINING-JOB.json" ]] && log "WARNING: a TRAINING-JOB appeared during drain"
  systemctl --user start affine-mimo-trainer-supervised.service
  ok=0; for i in $(seq 1 60); do "${OW[@]}" --status | grep -q '"alive": true' && { ok=1; break; }; sleep 5; done
  [[ $ok == 1 ]] || die "new trainer worker not alive"
  rm -f "$A/TRAINER-RESTART-REQUIRED"; log "TRAINER drain-restarted (cold optimizer rebuild under the current resolved config at the next job)";;
trainer)
  test -e "$UNITS/affine-mimo-lifecycle.service.d/$LC" || die "lifecycle v8.2.4 layer must be active first (v2 receipts)"
  pid=$(systemctl --user show affine-mimo-lifecycle.service -p MainPID --value)
  tr '\0' ' ' < /proc/$pid/cmdline | grep -q "release-v824-signed/uid-operator.v8.2.4" || die "live lifecycle is not on a v8.2.4 uid operator"
  [[ -e "$STATE/epochs/$EI/TRAINING-JOB.json" && ! -e "$STATE/epochs/$EI/TRAINING-RESULT.json" ]] && die "e$EI TRAINING-JOB in flight; never restart the trainer mid-job"
  OLD_SHA=$(grep -o 'owned-worker-v822s.json --sha256 [0-9a-f]*' "$UNITS/affine-mimo-trainer-supervised.service.d/zzzz~~~~v822s-owned-worker.conf" | awk '{print $3}')
  NEW_SHA=$("$PY" -c "import json;print(json.load(open('$A/PREPARE-RESULT.json'))['owned_worker_v824s_sha256'])")
  backup trainer
  # 1. host files + configs onto the pod (new directory; nothing existing overwritten)
  "${POD_SSH[@]}" "test ! -e $POD/host-v824 && mkdir -m 700 $POD/host-v824 $POD/host-v824/patched"
  (cd "$SG/trainer-host-v824" && tar -cf - .) | "${POD_SSH[@]}" "tar --no-same-owner -C $POD/host-v824 -xf - && chmod -R go-rwx $POD/host-v824"
  "${POD_SSH[@]}" "test ! -e $POD/owned-worker-v824s.json && umask 077 && cat > $POD/owned-worker-v824s.json" < "$A/pod/owned-worker-v824s.json"
  (cd "$SG/trainer-host-v824" && sha256sum $(find . -type f | sort)) > "$BK/pod-host-files.sha256"
  "${POD_SSH[@]}" "cd $POD/host-v824 && sha256sum -c --quiet" < "$BK/pod-host-files.sha256" || die "pod host files digest"
  [[ "$("${POD_SSH[@]}" "sha256sum $POD/owned-worker-v824s.json" | cut -c1-64)" == "$NEW_SHA" ]] || die "pod owned-worker-v824s digest"
  log "pod files installed; running owned-worker --check (launcher admission + overrides + worker --help; no model, no claims)"
  "${POD_SSH[@]}" "$POD/runtime-exact/venv/bin/python -I -B $POD/owned-worker-v7.py --config $POD/owned-worker-v824s.json --sha256 $NEW_SHA --check" 2>&1 | tail -3 | tee -a "$ALOG" | grep -q '"check": true' || die "owned-worker --check failed (nothing restarted)"
  # 2. drop-in, then drain-restart (worker idle between jobs)
  install -m 600 "$A/trainer/affine-mimo-trainer-supervised.service.d/$TC" "$UNITS/affine-mimo-trainer-supervised.service.d/$TC"
  systemctl --user daemon-reload
  [[ -e "$STATE/epochs/$(cur epoch_index)/TRAINING-JOB.json" && ! -e "$STATE/epochs/$(cur epoch_index)/TRAINING-RESULT.json" ]] && { rm -f "$UNITS/affine-mimo-trainer-supervised.service.d/$TC"; systemctl --user daemon-reload; die "job appeared; drop-in removed, nothing restarted"; }
  "${POD_SSH[@]}" "$POD/runtime-exact/venv/bin/python -I -B $POD/owned-worker-v7.py --config $POD/owned-worker-v822s.json --sha256 $OLD_SHA --stop-owned-after-drain" | tee -a "$ALOG"
  for i in $(seq 1 120); do
    "${POD_SSH[@]}" "$POD/runtime-exact/venv/bin/python -I -B $POD/owned-worker-v7.py --config $POD/owned-worker-v822s.json --sha256 $OLD_SHA --status" | grep -q '"alive": false' && break; sleep 5
  done
  systemctl --user restart affine-mimo-trainer-supervised.service; log "trainer supervised unit restarted on owned-worker-v824s"
  sleep 45
  "${POD_SSH[@]}" "$POD/runtime-exact/venv/bin/python -I -B $POD/owned-worker-v7.py --config $POD/owned-worker-v824s.json --sha256 $NEW_SHA --status" | tee -a "$ALOG" | grep -q '"alive": true' || die "v824 worker not alive (run rollback_v824.sh --trainer)"
  "${POD_SSH[@]}" "grep -a 'trainer-host-v824' $POD/owned-worker-v7.log | tail -1 | cut -c1-300" | tee -a "$ALOG"
  log "ACTIVATED trainer v8.2.4 (cold optimizer restore at the next job)";;
esac
