#!/usr/bin/bash
# SN120 prior-v1 BASELINE SWITCH (RA/20261101-baseline-switch; Jacob 2026-11-01 14:23-14:26 CR, pt1019u). Gate hooks (patch_gate.py:
# gate_service / training_params / gate_ledger / k1_transparency) + baseline_v1.py / prior_gate.py in the gate code dir + probe gate pool
# (probe-kit/install_gatepool.sh: envs.json 'affine_gatepool', probe_worker per-env gold) + PROMOTION-GATE.json heldout.baseline=prior-v1
# + llms.txt sections 5-7. The open ivw-v2 round is closed by the hook at the next tick (never revealed: it has no GREEN).
#   (default)    WAIT_EPOCH=<e> VL_WATCHDOG=stop|keep : wait for gate DECISION e (gap), take RA/LIFECYCLE-ACTIVATION.lock,
#                wait (<= 10 min, gate running) for an idle probe, install the probe kit, back up, patch code, flag on, texts, restart gate + 3
#                publishers, health check (else full revert), start the watchdog (unit affine-prior-watchdog).
#   MODE=revert  : stop the gate, close an open prior-v1 round (secret kept private in ROUND.private.json), reverse code, flag off,
#                texts back, restart. PROBE_REVERT=1 also restores the probe's envs.json/probe_worker.py (changes the probe fingerprint
#                again; the ivw-v2 path does not use affine_gatepool, so the default leaves it). WHY="..." for the log.
#   MODE=dryrun  : apply + revert on copies only (no live file touched), compared byte-exact with the live files.
set -u; export XDG_RUNTIME_DIR=/run/user/$(id -u) TZ=America/Costa_Rica
RA=/home/const/subnet120-rewrite/state/root-audits; B=$RA/20261101-baseline-switch; C=$B/code
GS=$RA/20261026-promotion-gate/state; FLAG=$RA/20261023-toploc-credit-train-gate/staged/PROMOTION-GATE.json
CODE=$RA/20261026-promotion-gate/code; K1=$RA/20261027-k1-per-uid/code/k1_transparency.py; WEB=/home/const/subnet120/affine/website
PY=/home/const/subnet120-rewrite/.venv/bin/python
LOGF=$B/state/activate.log; mkdir -p $B/state $B/backups; chmod 700 $B/backups
say(){ echo "$(date -u +%FT%TZ) $(date '+%T CR') $*" | tee -a $LOGF; }
prog(){ echo "- $(date '+%H:%M CR'): activate_baseline: $*" >> $B/PROGRESS.md; }
dec(){ printf '%s\t%s\t%s\t%s\n' "$(date '+%F %H:%M CR')" "$1" "$2" "$3" >> $B/DECISIONS.tsv; }
restart_all(){ systemctl --user restart affine-promotion-gate; systemctl --user restart affine-training-params affine-gate-ledger affine-k1-transparency; sleep 5; }
close_prior_round(){ $PY - "$1" <<'PY'
import json, os, sys, time
from pathlib import Path
H = Path('/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate/state/heldout')
try: cur = json.load(open(H / 'CURRENT.private.json'))
except Exception: print('no open round'); raise SystemExit
r = json.load(open(H / cur['id'] / 'ROUND.private.json'))
if r.get('baseline') != 'prior-v1': print('open round %s is an ivw-v2 round: kept' % cur['id']); raise SystemExit
p = H / cur['id'] / 'CLOSED.json'
if not p.exists():
    t = p.with_name('.CLOSED.json.tmp'); t.write_text(json.dumps(dict(round_id=cur['id'], closed_at=time.time(), closed_at_cr=time.strftime('%Y-%m-%d %H:%M:%S CR (UTC-6)'),
        reason='prior-v1 revert: %s' % sys.argv[1], closing_epoch=None), indent=1, sort_keys=True) + '\n'); os.chmod(t, 0o644); os.replace(t, p)
(H / 'CURRENT.private.json').unlink(missing_ok=True)
print('closed prior-v1 round %s' % cur['id'])
PY
}
revert(){
  WHY=$1
  [[ ${FROM_WATCHDOG:-} == 1 ]] || systemctl --user stop affine-prior-watchdog 2>/dev/null
  systemctl --user stop affine-promotion-gate; say "$(close_prior_round "$WHY")"
  $PY $C/patch_gate.py revert $CODE $K1 || { BK=$(cat $B/state/LAST-BACKUP); say "reverse patch failed: restoring backup copies $BK"; cp -p $BK/gate_service.py $BK/training_params.py $BK/gate_ledger.py $CODE/; cp -p $BK/k1_transparency.py $K1; }
  flock -w 60 $FLAG.lock python3 $C/setflag_prior.py $FLAG off "prior-v1 revert: $WHY"
  [[ -f $B/state/llms.pre-prior.txt ]] && python3 $C/patch_texts_prior.py revert x $WEB/llms.txt $B/state
  rm -f $WEB/gate/heldout/baseline_v1.py $WEB/gate/heldout/prior_gate.py
  [[ ${PROBE_REVERT:-} == 1 ]] && MODE=revert bash $B/probe-kit/install_gatepool.sh
  restart_all
  say "REVERT ($WHY): code reversed, flag ivw-v2, texts back; gate=$(systemctl --user is-active affine-promotion-gate) tp=$(systemctl --user is-active affine-training-params) ledger=$(systemctl --user is-active affine-gate-ledger) k1=$(systemctl --user is-active affine-k1-transparency)"
  prog "REVERT ($WHY)"; dec revert "$WHY" "code+flag+texts reversed"
  touch $B/state/REVERTED
}
if [[ ${MODE:-} == revert ]]; then revert "${WHY:-manual}"; exit 0; fi
if [[ ${MODE:-} == dryrun ]]; then
  D=$(mktemp -d /tmp/pv-dryrun.XXXX); mkdir -p $D/code $D/st
  for f in gate_service.py training_params.py gate_ledger.py heldout.py gate_core.py; do cp $CODE/$f $D/code/; done
  cp $K1 $D/k1_transparency.py; cp $WEB/llms.txt $D/llms.txt; cp $FLAG $D/PROMOTION-GATE.json
  cp $C/baseline_v1.py $C/prior_gate.py $D/code/
  $PY $C/patch_gate.py apply $D/code $D/k1_transparency.py >/dev/null && $PY $C/patch_gate.py apply $D/code $D/k1_transparency.py >/dev/null \
    && python3 $C/patch_texts_prior.py apply "$(date '+%Y-%m-%d %H:%M') CR" $D/llms.txt $D/st >/dev/null && python3 $C/setflag_prior.py $D/PROMOTION-GATE.json on dry >/dev/null \
    || { say "DRYRUN apply failed"; exit 1; }
  (cd $D/code && $PY -B -c "import sys; sys.path.insert(0, '.'); import ast; [ast.parse(open(f).read()) for f in ('gate_service.py', 'training_params.py', 'gate_ledger.py', '../k1_transparency.py')]; import baseline_v1, prior_gate; print('import ok')") || { say "DRYRUN import failed"; exit 1; }
  $PY $C/patch_gate.py revert $D/code $D/k1_transparency.py >/dev/null && python3 $C/patch_texts_prior.py revert x $D/llms.txt $D/st >/dev/null \
    && python3 $C/setflag_prior.py $D/PROMOTION-GATE.json off dry >/dev/null || { say "DRYRUN revert failed"; exit 1; }
  ok=1; for f in gate_service.py training_params.py gate_ledger.py; do cmp -s $CODE/$f $D/code/$f || { say "DRYRUN mismatch $f"; ok=0; }; done
  cmp -s $K1 $D/k1_transparency.py || { say "DRYRUN mismatch k1_transparency.py"; ok=0; }; cmp -s $WEB/llms.txt $D/llms.txt || { say "DRYRUN mismatch llms.txt"; ok=0; }
  python3 - $FLAG $D/PROMOTION-GATE.json <<'PY' || ok=0
import json, sys
a, b = (json.load(open(x))['heldout'] for x in sys.argv[1:]); b.pop('prior_v1_history', None); a.pop('prior_v1_history', None)
diff = {k: (a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)}
print('flag diff after on+off (besides history):', diff or 'none'); sys.exit(1 if diff else 0)
PY
  say "DRYRUN apply+revert: $([[ $ok == 1 ]] && echo OK byte-exact || echo FAILED) (copies in $D)"; exit $((1 - ok))
fi
W=${WAIT_EPOCH:?set WAIT_EPOCH}; VLW=${VL_WATCHDOG:?set VL_WATCHDOG=stop or keep (the variance-levers watchdog reverts the levers on generic gate errors)}
python3 -c "import json,sys; h=json.load(open('$FLAG'))['heldout']; sys.exit(0 if h.get('enabled') is True and h.get('baseline') in (None, 'ivw-v2') else 1)" \
  || { say "ABORT: heldout not enabled or baseline already switched"; exit 1; }
[[ -f $B/state/REVERTED ]] && { say "ABORT: REVERTED marker present (remove it to re-arm)"; exit 1; }
$PY $C/patch_gate.py check $CODE $K1 || { say "ABORT: patch anchors missing in live code"; exit 1; }
MODE=dryrun bash $0 || { say "ABORT: dry run failed"; exit 1; }
MODE=check bash $B/probe-kit/install_gatepool.sh >/dev/null 2>&1; true
systemctl --user is-active affine-prior-table >/dev/null || say "WARN: affine-prior-table not active (the round uses the table as of its last write)"
$PY -c "import sqlite3; c=sqlite3.connect('file:$RA/20261101-prior-table/private/table/prior_table.private.sqlite?mode=ro', uri=True); print('table rows', c.execute('select count(*) from tasks').fetchone()[0])" || { say "ABORT: prior table unreadable"; exit 1; }
sha256sum -c <(echo "9e05d47c4672c8e066a15079577334f0432898ea06afc7045c82862345465a8e  $B/public/gate-pool-v1.jsonl") >/dev/null || { say "ABORT: gate pool sha256"; exit 1; }
SF=$RA/20261101-grade-stall-fix
python3 - $SF <<'PY' || { say "ABORT: grade-stall fix not verified (needs state/VERIFIED, or its first round's baseline done + >= 1 decided turn, no REVERTED)"; exit 1; }
import json, sys
from pathlib import Path
sf = Path(sys.argv[1]); st = sf / 'state'
if (st / 'REVERTED').exists() or not (st / 'INSTALLED.json').exists(): sys.exit(1)
if (st / 'VERIFIED').exists(): print('stall fix VERIFIED'); sys.exit(0)
g = json.load(open(st / 'GUARD-STATUS.json')); r = (g.get('new_rounds') or [{}])[0]
ok = bool(r.get('baseline_s')) and any(t[1] in ('red', 'green') and not t[3] for t in r.get('turns') or []) and not g.get('errors')
print('stall fix first round %s baseline %ss turns %s errors %s' % (r.get('id'), r.get('baseline_s'), r.get('turns'), g.get('errors'))); sys.exit(0 if ok else 1)
PY
say "staged ok; waiting for gate DECISION e$W"; prog "armed: waiting for gate DECISION e$W"
while :; do
  until [[ -f $GS/epochs/$W/DECISION.json ]]; do sleep 2; done; sleep 1
  exec 9>$RA/LIFECYCLE-ACTIVATION.lock
  if flock -n 9; then break; fi
  exec 9>&-; say "LIFECYCLE-ACTIVATION.lock held by another job after e$W; waiting for the next gap"; W=$((W + 1))
done
TS=$(date -u +%Y%m%dT%H%M%SZ); BK=$B/backups/pv-$TS; mkdir -m 700 -p $BK; T="$(date '+%Y-%m-%d %H:%M') CR"
cp -a $FLAG $CODE/heldout.py $CODE/gate_service.py $CODE/gate_core.py $CODE/training_params.py $CODE/gate_ledger.py $K1 $WEB/llms.txt $WEB/mining.json $WEB/gate.json $BK/ \
  && echo $BK > $B/state/LAST-BACKUP || { say "ABORT backup"; exit 1; }
echo "$T" > $B/state/SWITCH_CR; echo $((W + 1)) > $B/state/SWITCH_EPOCH
# the gate keeps running (the grade-stall-fix guard reverts if the gate is inactive >= 60 s): install in the post-DECISION gap
# when the probe is idle, then patch + restart right away (seconds), so no run starts on mixed files.
for i in $(seq 1 20); do MODE=check bash $B/probe-kit/install_gatepool.sh 2>&1 | grep -q '^active runs=\[\] fetching=False' && break; [[ $i == 20 ]] && { say "ABORT: probe not idle after 10 min"; exit 1; }; sleep 30; done
MODE=install bash $B/probe-kit/install_gatepool.sh || { say "ABORT: probe kit install failed (pod restored)"; exit 1; }
cp -p $C/baseline_v1.py $C/prior_gate.py $CODE/ && $PY $C/patch_gate.py apply $CODE $K1 || { say "ABORT: code patch failed"; cp -p $BK/gate_service.py $BK/training_params.py $BK/gate_ledger.py $CODE/; cp -p $BK/k1_transparency.py $K1; MODE=revert bash $B/probe-kit/install_gatepool.sh; exit 1; }
flock -w 60 $FLAG.lock python3 $C/setflag_prior.py $FLAG on "prior-v1 activation after e$W DECISION"
python3 $C/patch_texts_prior.py apply "$T" $WEB/llms.txt $B/state || say "WARN: text patch rc!=0"
mkdir -p $WEB/gate/heldout && cp -p $C/baseline_v1.py $C/prior_gate.py $WEB/gate/heldout/
[[ $VLW == stop ]] && { systemctl --user stop affine-vl-watchdog 2>/dev/null; say "variance-levers watchdog stopped (prior-v1 watchdog takes over)"; }
T0=$(date +%s); restart_all
say "INSTALLED after e$W DECISION (backup $BK): probe gate pool, code patched, flag prior-v1, texts; services restarted"
prog "INSTALLED after e$W DECISION; backup $BK; switch $T"; dec install "after e$W DECISION" "backup $BK"
exec 9>&-
sleep 60
A=$(systemctl --user is-active affine-promotion-gate); P=$(systemctl --user is-active affine-training-params); L=$(systemctl --user is-active affine-gate-ledger); K=$(systemctl --user is-active affine-k1-transparency)
ERR=$(python3 - $T0 <<'PY'
import json, sys
t0 = float(sys.argv[1]); n = 0; ticks = 0; ev = []
for l in open('/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate/state/gate.log', errors='replace').readlines()[-400:]:
    try: d = json.loads(l)
    except Exception: n += 'Traceback' in l; continue
    if d.get('at', 0) < t0: continue
    n += 'error' in d.get('event', ''); ticks += d.get('event') == 'tick'; ev += [d['event']] if str(d.get('event', '')).startswith('prior_') else []
print('%d %d %s' % (n, ticks, ','.join(ev) or '-'))
PY
)
if [[ $A != active || $P != active || $L != active || $K != active || ${ERR%% *} != 0 ]]; then revert "post-install health: gate=$A tp=$P ledger=$L k1=$K errors/ticks/prior-events=$ERR"; exit 1; fi
say "LIVE gate=$A tp=$P ledger=$L k1=$K errors/ticks/prior-events=$ERR"
prog "LIVE (gate=$A tp=$P ledger=$L k1=$K errors/ticks/prior-events=$ERR)"
systemd-run --user --unit=affine-prior-watchdog --description="SN120 prior-v1 baseline watchdog (auto-revert)" $PY -u $C/watchdog_prior.py $T0 \
  && say "watchdog started (unit affine-prior-watchdog)" || say "WARN: watchdog failed to start"
