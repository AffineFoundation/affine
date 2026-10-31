#!/usr/bin/bash
# SN120 VARIANCE LEVERS (RA/20261031-variance-levers; Jacob 2026-10-31 14:07 CR: pooled sigma_run, 8 candidate rollouts per task,
# 2,000 tasks per round). Gate code patch (patch_levers.py: k_cand + sigma_run policy frozen per round in ROUND.private.json) +
# PROMOTION-GATE.json heldout n 2000 / k_cand 8 / sigma_run_policy pooled. The OPEN round keeps its own rules (k_cand 4 from its
# record, per-round sigma_run); only rounds opened after the switch use the new values. Bar unchanged (gain > 0, z > 1.00, cap < 0.20).
#   (default)       : wait for the next gate DECISION (gap), take RA/LIFECYCLE-ACTIVATION.lock (non-blocking), back up, patch code,
#                     flag on, texts, restart gate + 3 publishers, health check (else full revert), start the watchdog.
#   MODE=revert     : reverse code + flag off + texts back + restart; a still-open NEW-rules round is closed (never revealed) so the
#                     next round opens under the old rules (NOCLOSE=1 to keep it). WHY="..." for the log.
#   MODE=dryrun     : revert path on copies only (no live file touched), compared byte-exact with the backups.
set -u; export XDG_RUNTIME_DIR=/run/user/$(id -u) TZ=America/Costa_Rica
RA=/home/const/subnet120-rewrite/state/root-audits; VL=$RA/20261031-variance-levers
GS=$RA/20261026-promotion-gate/state; FLAG=$RA/20261023-toploc-credit-train-gate/staged/PROMOTION-GATE.json
CODE=$RA/20261026-promotion-gate/code; K1=$RA/20261027-k1-per-uid/code/k1_transparency.py; WEB=/home/const/subnet120/affine/website
PY=/home/const/subnet120-rewrite/.venv/bin/python
LOGF=$VL/state/activate.log; mkdir -p $VL/state $VL/backups
say(){ echo "$(date -u +%FT%TZ) $(date '+%T CR') $*" | tee -a $LOGF; }
prog(){ echo "- $(date '+%H:%M CR'): activate_levers: $*" >> $VL/PROGRESS.md; }
dec(){ printf '%s\t%s\t%s\t%s\n' "$(date '+%F %H:%M CR')" "$1" "$2" "$3" >> $VL/DECISIONS.tsv; }
TEXTS="$CODE/training_params.py $CODE/gate_ledger.py $K1 $WEB/llms.txt"
restart_all(){ systemctl --user restart affine-promotion-gate; systemctl --user restart affine-training-params affine-gate-ledger affine-k1-transparency; sleep 5; }
close_new_round(){ $PY - "$1" <<'PY'
import json, os, sys, time
from pathlib import Path
H = Path('/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate/state/heldout')
try: cur = json.load(open(H / 'CURRENT.private.json'))
except Exception: print('no open round'); raise SystemExit
r = json.load(open(H / cur['id'] / 'ROUND.private.json'))
if not isinstance(r.get('sigma_run_policy'), dict) and int(r.get('k_cand', 4)) == 4 and int(r.get('n', 1000)) == 1000:
    print('open round %s is an old-rules round: kept' % cur['id']); raise SystemExit
p = H / cur['id'] / 'CLOSED.json'
if not p.exists():
    t = p.with_name('.CLOSED.json.tmp'); t.write_text(json.dumps(dict(round_id=cur['id'], closed_at=time.time(), closed_at_cr=time.strftime('%Y-%m-%d %H:%M:%S CR (UTC-6)'),
        reason='variance-levers revert: %s' % sys.argv[1], closing_epoch=None), indent=1, sort_keys=True) + '\n'); os.chmod(t, 0o644); os.replace(t, p)
(H / 'CURRENT.private.json').unlink(missing_ok=True)
print('closed new-rules round %s (never revealed)' % cur['id'])
PY
}
revert(){
  WHY=$1; T=$(cat $VL/state/SWITCH_CR 2>/dev/null || echo "?")
  [[ ${NOCLOSE:-} == 1 ]] || { systemctl --user stop affine-promotion-gate; say "$(close_new_round "$WHY")"; }
  $PY $VL/patch_levers.py revert $CODE/heldout.py $CODE/gate_service.py || { BK=$(cat $VL/state/LAST-BACKUP); say "reverse patch failed: restoring backup copies $BK"; cp -p $BK/heldout.py $BK/gate_service.py $CODE/; }
  flock -w 60 $FLAG.lock python3 $VL/setflag.py $FLAG off "variance levers revert: $WHY"
  [[ $T != "?" ]] && python3 $VL/patch_texts.py revert "$T" $TEXTS
  cp -p $CODE/heldout.py $WEB/gate/heldout/heldout.py
  restart_all
  say "REVERT ($WHY): code reversed, flag n 1000 / k_cand 4 / per-round sigma_run, texts back; gate=$(systemctl --user is-active affine-promotion-gate) tp=$(systemctl --user is-active affine-training-params) ledger=$(systemctl --user is-active affine-gate-ledger) k1=$(systemctl --user is-active affine-k1-transparency)"
  prog "REVERT ($WHY)"; dec revert "$WHY" "code+flag+texts reversed"
  touch $VL/state/REVERTED
}
if [[ ${MODE:-} == revert ]]; then revert "${WHY:-manual}"; exit 0; fi
if [[ ${MODE:-} == dryrun ]]; then
  BK=$(cat $VL/state/LAST-BACKUP 2>/dev/null || ls -d $VL/backups/pre-stage-* | tail -1); D=$(mktemp -d /tmp/vl-dryrun.XXXX); T=$(cat $VL/state/SWITCH_CR 2>/dev/null || echo "2026-10-31 14:30 CR")
  for f in heldout.py gate_service.py training_params.py gate_ledger.py k1_transparency.py llms.txt PROMOTION-GATE.json; do cp $BK/$f $D/; done
  $PY $VL/patch_levers.py apply $D/heldout.py $D/gate_service.py >/dev/null && python3 $VL/patch_texts.py apply "$T" $D/training_params.py $D/gate_ledger.py $D/k1_transparency.py $D/llms.txt >/dev/null \
    && python3 $VL/setflag.py $D/PROMOTION-GATE.json on dry >/dev/null || { say "DRYRUN apply failed"; exit 1; }
  $PY $VL/patch_levers.py revert $D/heldout.py $D/gate_service.py >/dev/null && python3 $VL/patch_texts.py revert "$T" $D/training_params.py $D/gate_ledger.py $D/k1_transparency.py $D/llms.txt >/dev/null \
    && python3 $VL/setflag.py $D/PROMOTION-GATE.json off dry >/dev/null || { say "DRYRUN revert failed"; exit 1; }
  ok=1; for f in heldout.py gate_service.py training_params.py gate_ledger.py k1_transparency.py llms.txt; do cmp -s $BK/$f $D/$f || { say "DRYRUN mismatch $f"; ok=0; }; done
  python3 - $BK/PROMOTION-GATE.json $D/PROMOTION-GATE.json <<'PY' || ok=0
import json, sys
a, b = (json.load(open(x))['heldout'] for x in sys.argv[1:]); b.pop('variance_levers_history', None)
diff = {k: (a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k) and k != 'sigma_run_policy'}
print('flag diff after on+off (besides history, sigma_run_policy=per-round):', diff or 'none'); sys.exit(1 if diff else 0)
PY
  say "DRYRUN revert path: $([[ $ok == 1 ]] && echo OK byte-exact || echo FAILED) (copies in $D)"; exit $((1 - ok))
fi
W=${WAIT_EPOCH:?set WAIT_EPOCH}
python3 -c "import json,sys; h=json.load(open('$FLAG'))['heldout']; sys.exit(0 if h.get('enabled') is True and int(h['n'])==1000 and int(h['k_cand'])==4 and h.get('sigma_run_policy','per-round')=='per-round' else 1)" \
  || { say "ABORT: heldout not enabled or levers already set"; exit 1; }
MODE=dryrun bash $0 || { say "ABORT: dry run failed"; exit 1; }
say "staged ok; waiting for gate DECISION e$W"; prog "armed: waiting for gate DECISION e$W"
while :; do
  until [[ -f $GS/epochs/$W/DECISION.json ]]; do sleep 2; done; sleep 1
  exec 9>$RA/LIFECYCLE-ACTIVATION.lock
  if flock -n 9; then break; fi
  exec 9>&-; say "LIFECYCLE-ACTIVATION.lock held by another job after e$W; waiting for the next gap"; W=$((W + 1))
done
TS=$(date -u +%Y%m%dT%H%M%SZ); BK=$VL/backups/vl-$TS; mkdir -m 700 -p $BK; T="$(date '+%Y-%m-%d %H:%M') CR"
cp -a $FLAG $CODE/heldout.py $CODE/gate_service.py $CODE/gate_core.py $CODE/training_params.py $CODE/gate_ledger.py $K1 $WEB/llms.txt $WEB/mining.json $WEB/gate.json $BK/ \
  && cp -a $WEB/gate/heldout/heldout.py $BK/site-heldout.py && echo $BK > $VL/state/LAST-BACKUP || { say "ABORT backup"; exit 1; }
echo "$T" > $VL/state/SWITCH_CR; echo $((W + 1)) > $VL/state/SWITCH_EPOCH
$PY $VL/patch_levers.py apply $CODE/heldout.py $CODE/gate_service.py || { say "ABORT: code patch failed (nothing changed)"; cp -p $BK/heldout.py $BK/gate_service.py $CODE/; exit 1; }
flock -w 60 $FLAG.lock python3 $VL/setflag.py $FLAG on "variance levers activation after e$W DECISION"
python3 $VL/patch_texts.py apply "$T" $TEXTS || say "WARN: text patch rc!=0"
cp -p $CODE/heldout.py $WEB/gate/heldout/heldout.py
T0=$(date +%s); restart_all
say "INSTALLED after e$W DECISION (backup $BK): code patched, flag n 2000 / k_cand 8 / pooled sigma_run for rounds opened after $T; texts; services restarted"
prog "INSTALLED after e$W DECISION; backup $BK; switch $T"; dec install "after e$W DECISION" "backup $BK"
exec 9>&-
sleep 40
A=$(systemctl --user is-active affine-promotion-gate); P=$(systemctl --user is-active affine-training-params); L=$(systemctl --user is-active affine-gate-ledger); K=$(systemctl --user is-active affine-k1-transparency)
ERR=$(python3 - $T0 <<'PY'
import json, sys
t0 = float(sys.argv[1]); n = 0; ticks = 0
for l in open('/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate/state/gate.log', errors='replace').readlines()[-400:]:
    try: d = json.loads(l)
    except Exception: continue
    if d.get('at', 0) < t0: continue
    n += 'error' in d.get('event', ''); ticks += d.get('event') == 'tick'
print('%d %d' % (n, ticks))
PY
)
if [[ $A != active || $P != active || $L != active || $K != active || ${ERR%% *} != 0 ]]; then revert "post-install health: gate=$A tp=$P ledger=$L k1=$K errors/ticks=$ERR"; exit 1; fi
say "LIVE gate=$A tp=$P ledger=$L k1=$K errors/ticks=$ERR; next round opens with n 2000, k_cand 8, pooled sigma_run"
prog "LIVE (gate=$A tp=$P ledger=$L k1=$K errors/ticks=$ERR)"
systemd-run --user --unit=affine-vl-watchdog --description="SN120 variance-levers watchdog (auto-revert)" $PY -u $VL/watchdog.py $T0 \
  && say "watchdog started (unit affine-vl-watchdog)" || say "WARN: watchdog failed to start"
