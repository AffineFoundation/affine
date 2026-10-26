#!/usr/bin/env bash
# Activate v829h collection-window release on the lifecycle (Jacob 2026-10-26 13:38 CR "Cut gather to 20 minutes").
# Waits for e(N) FINALIZED(updated) + e(N+1) collecting (PLAN, no dispatch/job) with >=10 min left, and for the
# concurrent v8.3.2 trainer activation to finish; then backs up, rewrites the v829 lifecycle drop-in, restarts.
# The window switches at the NEXT boundary (e(N+2) is the first epoch under the new window).
# usage: REL=release-v829h-collect1200-signed FROM_EPOCH=150 activate_v829h.sh
# revert: REL=release-v829h-collect1800-signed FROM_EPOCH=<finalized epoch> activate_v829h.sh
set -euo pipefail; umask 077
AUDIT=/home/const/subnet120-rewrite/state/root-audits/20261023-toploc-credit-train-gate
REL=$AUDIT/${REL:?REL}
N0=${FROM_EPOCH:?FROM_EPOCH}; N1=$((N0+1))
STATE=/home/const/subnet120-rewrite/state/mimo-mainnet-v1; E=$STATE/epochs
UNITS=/home/const/.config/systemd/user; LD=$UNITS/affine-mimo-lifecycle.service.d
LC=zzzz~~~~~~~~~~~~~v829-train-gate.conf; SVC=affine-mimo-lifecycle.service
PY=/home/const/subnet120-rewrite/.venv/bin/python
V828OP=/home/const/subnet120-rewrite/state/root-audits/20261021-v828-audit-off-path/release-v828-signed/v828-operator.ROOT-ADMITTED.private.json
V832LOG=/home/const/subnet120-rewrite/state/root-audits/20261026-gradnorm/release/ACTIVATION-v832.log
LOG=$AUDIT/activate-v829h.log
cr(){ TZ=America/Costa_Rica date "+%Y-%m-%d %I:%M:%S %p CR"; }
log(){ echo "$(date -u +%FT%TZ) $(cr) $*" | tee -a "$LOG"; }
die(){ log "ABORT: $*"; exit 1; }
(cd "$REL" && sha256sum --quiet -c SIGNED-MANIFEST.sha256) || die "release manifest"
python3 -c "import json;assert json.load(open('$REL/SIGNING-RESULT.json'))['checks_passed'] is True" || die "signing result"
log "armed REL=$(basename $REL) FROM_EPOCH=$N0; waiting for e$N0 FINALIZED + e$N1 collecting + v832 activation done"
while true; do
  if [[ -e $E/$N0/RECOVERY-REQUIRED.json ]]; then die "e$N0 RECOVERY-REQUIRED"; fi
  if [[ -f $E/$N0/FINALIZED.json && -f $E/$N1/PLAN.json ]]; then
    for f in TRAINING-JOB.json FINALIZED.json AUDIT-POPULATION.json TRAIN-DISPATCH-V829C.json coordinator/DISPATCH.json coordinator/POPULATION.json; do
      if [[ -e $E/$N1/$f ]]; then die "e$N1 already past collection ($f); rerun with FROM_EPOCH=$N1"; fi
    done
    left=$($PY -c "import json,time;p=json.load(open('$E/$N1/PLAN.json'));p=p.get('payload',p);print(int(p['manifest']['deadline']-time.time()))")
    if (( left < 600 )); then die "only ${left}s to e$N1 deadline; rerun with FROM_EPOCH=$N1"; fi
    v832_busy=0
    if pgrep -f '[a]ctivate_v832.sh' >/dev/null; then v832_busy=1; fi
    if [[ $v832_busy == 0 ]]; then break; fi
    log "e$N1 collecting (${left}s left); v832 trainer activation still running - waiting"
  fi
  sleep 15
done
grep -q '"updated"' $E/$N0/FINALIZED.json || die "e$N0 FINALIZED without training=updated"
log "safe window: e$N0 FINALIZED, e$N1 collecting ${left}s left, v832 activation not running ($(tail -1 $V832LOG 2>/dev/null | cut -c1-160))"
$PY -B "$REL/mimo_v829_train_gate_launcher.py" --operator-config "$REL/v829-operator.ROOT-ADMITTED.private.json" --v828-operator-config "$V828OP" --check > /tmp/v829h-check.json
python3 -c "import json;d=json.load(open('/tmp/v829h-check.json'));assert d['v829_check'] is True and d['v829']['v829h']['v829h'] is True;print(d['v829']['v829h'])" | tee -a "$LOG"
TS=$(date -u +%Y%m%dT%H%M%SZ); BK=$AUDIT/backup-lifecycle-v829h-$TS; mkdir -m 700 "$BK"; echo "$BK" > $AUDIT/LAST-BACKUP-v829h
tar -C "$UNITS" -cf "$BK/systemd-user-units.tar" affine-mimo-lifecycle.service.d
cp -a $STATE/CURRENT.json "$BK/CURRENT.json.pre"
cp -a $AUDIT/observer.v8.2.9.private.json "$BK/observer.v8.2.9.private.json.pre"
systemctl --user show $SVC -p ExecStart --value > "$BK/lifecycle-execstart.pre.txt"
sha256sum "$BK"/* > "$BK/BACKUP.sha256"
log "backups -> $BK"
mkdir -p -m 700 "$AUDIT/activation-v829h"
cat > "$AUDIT/activation-v829h/$LC" << EOC
# MiMo host v8.2.9h: v829g unchanged + collection_seconds boundary switch ($(basename $REL)). Jacob 2026-10-26 13:38 CR "Cut gather to 20 minutes".
# Installed by $AUDIT/activate_v829h.sh $(cr). Revert: REL=release-v829h-collect1800-signed FROM_EPOCH=<finalized> activate_v829h.sh
[Service]
ExecStart=
ExecStart=$PY -B $REL/mimo_v829_train_gate_launcher.py --operator-config $REL/v829-operator.ROOT-ADMITTED.private.json --v828-operator-config $V828OP
EOC
chmod 600 "$AUDIT/activation-v829h/$LC"
install -m 600 "$AUDIT/activation-v829h/$LC" "$LD/$LC"
systemctl --user daemon-reload
systemctl --user show $SVC -p ExecStart --value | grep -q "$REL/mimo_v829_train_gate_launcher.py" || { tar -C "$UNITS" -xf "$BK/systemd-user-units.tar"; systemctl --user daemon-reload; die "effective ExecStart not v829h (restored)"; }
systemctl --user restart $SVC
log "lifecycle restarted on $(basename $REL)"
"$AUDIT/rebind_mining_observer_to_release.sh" "$REL" 2>&1 | tee -a "$LOG" || log "WARNING: observer rebind failed (follow-live still applies)"
sleep 45
systemctl --user is-active --quiet $SVC || die "lifecycle not active - restore $BK/systemd-user-units.tar"
pid=$(systemctl --user show $SVC -p MainPID --value)
tr '\0' ' ' < /proc/$pid/cmdline | grep -q "$(basename $REL)" || die "lifecycle pid $pid not on $(basename $REL)"
journalctl --user -u $SVC --since '-2min' --no-pager 2>/dev/null | grep -E 'host-v829|Traceback|Error' | cut -c1-400 | tail -8 | tee -a "$LOG" || true
$PY -c "import json;d=json.load(open('$STATE/CURRENT.json'));print('CURRENT',{k:d[k] for k in ('epoch_index','collection_seconds','max_tasks')})" | tee -a "$LOG"
log "ACTIVATED $(basename $REL); first epoch under the new window = e$((N1+1))"
