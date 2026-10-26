#!/usr/bin/env bash
# Activate lifecycle release v8.2.9i (promotion gate) and switch the gate flag ON. Jacob pt682u/t684u 2026-10-26.
# Waits until: CURRENT epoch >= FROM_EPOCH is collecting (PLAN, no DISPATCH/JOB) with >= MINLEFT s left, the rollback2
# auto_continue has finished (TRAIN-CONTROL mode=train), no activate_* script running. Then backup, drop-in, restart,
# observer rebind, flag ON. The collecting epoch is the first gated epoch.
# Revert (no restart): flag enabled=false  ->  exact v829h behaviour.  Full revert: REVERT=1 activate_v829i.sh
set -euo pipefail; umask 077
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
AUDIT=/home/const/subnet120-rewrite/state/root-audits/20261023-toploc-credit-train-gate
V=/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate
REL=$AUDIT/release-v829i-gate-signed; [[ "${REVERT:-0}" == 1 ]] && REL=$AUDIT/release-v829h-collect1200-signed
FROM=${FROM_EPOCH:?FROM_EPOCH}; MINLEFT=${MINLEFT:-420}
STATE=/home/const/subnet120-rewrite/state/mimo-mainnet-v1; E=$STATE/epochs
UNITS=/home/const/.config/systemd/user; LD=$UNITS/affine-mimo-lifecycle.service.d
LC=zzzz~~~~~~~~~~~~~v829-train-gate.conf; SVC=affine-mimo-lifecycle.service
PY=/home/const/subnet120-rewrite/.venv/bin/python
V828OP=/home/const/subnet120-rewrite/state/root-audits/20261021-v828-audit-off-path/release-v828-signed/v828-operator.ROOT-ADMITTED.private.json
FLAG=$AUDIT/staged/PROMOTION-GATE.json; CTL=$AUDIT/staged/TRAIN-CONTROL.json
LOG=$V/state/activate-v829i.log
cr(){ TZ=America/Costa_Rica date "+%Y-%m-%d %I:%M:%S %p CR"; }
log(){ echo "$(date -u +%FT%TZ) $(cr) $*" | tee -a "$LOG"; }
die(){ log "ABORT: $*"; exit 1; }
(cd "$REL" && sha256sum --quiet -c SIGNED-MANIFEST.sha256) || die "release manifest"
python3 -c "import json;assert json.load(open('$REL/SIGNING-RESULT.json'))['checks_passed'] is True" || die "signing result"
log "armed REL=$(basename $REL) FROM_EPOCH=$FROM MINLEFT=$MINLEFT"
while true; do
  N=$($PY -c "import json;print(json.load(open('$STATE/CURRENT.json'))['epoch_index'])")
  if (( N >= FROM )) && [[ -f $E/$N/PLAN.json && ! -e $E/$N/coordinator/DISPATCH.json && ! -e $E/$N/TRAINING-JOB.json ]]; then
    left=$($PY -c "import json,time;p=json.load(open('$E/$N/PLAN.json'));p=p.get('payload',p);print(int(p['manifest']['deadline']-time.time()))")
    mode=$($PY -c "import json;print(json.load(open('$CTL')).get('mode'))" 2>/dev/null || echo absent)
    busy=0; pgrep -f '[a]uto_continue_v832.sh' >/dev/null && busy=1; pgrep -f '[a]ctivate_v832.sh|[a]ctivate_v829h.sh|[s]top_and_restore' >/dev/null && busy=1 || true
    if (( left < MINLEFT )); then log "e$N only ${left}s left; waiting for next epoch"; FROM=$((N+1));
    elif [[ $busy == 0 && ( $mode == train || $mode == absent ) ]]; then break;
    else log "e$N collecting ${left}s left; TRAIN-CONTROL=$mode busy=$busy - waiting"; fi
  fi
  sleep 15
done
log "safe window: e$N collecting ${left}s left, TRAIN-CONTROL=$mode"
$PY -B "$REL/mimo_v829_train_gate_launcher.py" --operator-config "$REL/v829-operator.ROOT-ADMITTED.private.json" --v828-operator-config "$V828OP" --check > /tmp/v829i-check.json
python3 -c "import json;d=json.load(open('/tmp/v829i-check.json'));assert d['v829_check'] is True;print(d['v829'].get('v829i'), d['v829']['v829h'])" | tee -a "$LOG"
TS=$(date -u +%Y%m%dT%H%M%SZ); BK=$V/state/backup-lifecycle-v829i-$TS; mkdir -m 700 "$BK"; echo "$BK" > $V/state/LAST-BACKUP-v829i
tar -C "$UNITS" -cf "$BK/systemd-user-units.tar" affine-mimo-lifecycle.service.d
cp -a $STATE/CURRENT.json "$BK/CURRENT.json.pre"; cp -a $AUDIT/observer.v8.2.9.private.json "$BK/observer.v8.2.9.private.json.pre"
[[ -f $FLAG ]] && cp -a $FLAG "$BK/PROMOTION-GATE.json.pre"
systemctl --user show $SVC -p ExecStart --value > "$BK/lifecycle-execstart.pre.txt"; sha256sum "$BK"/* > "$BK/BACKUP.sha256"
log "backups -> $BK"
cat > "$V/state/$LC" << EOC
# MiMo host $(basename $REL): v829h (collect 1200) + promotion gate wrapper v8.2.9i (flag staged/PROMOTION-GATE.json). Jacob pt682u/t684u 2026-10-26.
# Installed by $V/code/activate_v829i.sh $(cr). Revert: flag enabled=false (no restart) or REVERT=1 FROM_EPOCH=<N> activate_v829i.sh
[Service]
ExecStart=
ExecStart=$PY -B $REL/mimo_v829_train_gate_launcher.py --operator-config $REL/v829-operator.ROOT-ADMITTED.private.json --v828-operator-config $V828OP
EOC
install -m 600 "$V/state/$LC" "$LD/$LC"; systemctl --user daemon-reload
systemctl --user show $SVC -p ExecStart --value | grep -q "$REL/mimo_v829_train_gate_launcher.py" || { tar -C "$UNITS" -xf "$BK/systemd-user-units.tar"; systemctl --user daemon-reload; die "effective ExecStart wrong (restored)"; }
if [[ "${REVERT:-0}" == 1 ]]; then
  python3 -c "import json;p='$FLAG';d=json.load(open(p));d['enabled']=False;open(p,'w').write(json.dumps(d,indent=1,sort_keys=True)+'\n')" 2>/dev/null || true
else
  python3 - <<PY
import json, os
p = "$FLAG"
d = dict(enabled=True, shadow=True, sample_size=1000, z_min=-1.0, cap_max=0.20, min_tasks=200, red_credit="zero", pool=7496, env="affine_math",
         decision_timeout_seconds=3600, restore_wait_seconds=1200, h_wait_seconds=2400,
         authorized_by=["pt682u", "t684u"], note="Jacob 2026-10-26: promotion gate LIVE. enabled=false reverts to v829h behaviour without a restart.",
         first_gated_epoch=$N, written_at_cr="$(cr)")
t = p + ".tmp"; open(t, "w").write(json.dumps(d, indent=1, sort_keys=True) + "\n"); os.chmod(t, 0o600); os.replace(t, p)
PY
fi
systemctl --user restart $SVC
log "lifecycle restarted on $(basename $REL); flag: $(cat $FLAG | tr -d '\n' | cut -c1-120)"
"$AUDIT/rebind_mining_observer_to_release.sh" "$REL" 2>&1 | tee -a "$LOG" || log "WARNING: observer rebind failed"
sleep 45
systemctl --user is-active --quiet $SVC || die "lifecycle not active - restore $BK/systemd-user-units.tar"
pid=$(systemctl --user show $SVC -p MainPID --value)
tr '\0' ' ' < /proc/$pid/cmdline | grep -q "$(basename $REL)" || die "lifecycle pid $pid not on $(basename $REL)"
journalctl --user -u $SVC --since '-2min' --no-pager 2>/dev/null | grep -E 'host-v829|Traceback|Error' | cut -c1-300 | tail -6 | tee -a "$LOG" || true
log "ACTIVATED $(basename $REL); first gated epoch = e$N"
