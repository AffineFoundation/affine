#!/usr/bin/bash
# SN120 z bar step-down (RA/20261101-zbar-stepdown; Jacob 2026-11-01 21:23 + 21:25 CR): GREEN needs z > z_bar, z_bar = 1.00 / 0.50 after 6 h /
# 0.25 after 12 h since the last GREEN (clock: candidate TRAINING-RESULT time - last GREEN decided_at), reset on GREEN.
# Files: CODE/baseline_v1.py + prior_gate.py + gate_ledger.py (+ public copies WEB/gate/heldout), WEB/k1.js (+ index.html cache tag),
# PROMOTION-GATE.json heldout.prior_v1 z_bar_*, llms.txt; publishes WEB/gate/zbar.json and gate.json z_bar.
#   (default) AFTER_EPOCH=<e>: wait for DECISION e and a quiet gap (no TRAINING-RESULT e+1, gate not busy; else the next gap), back up,
#             install, restart gate + 3 publishers, 60 s health check (else revert), hand the cap watchdog's job to affine-zbar-watchdog.
#   MODE=revert WHY="...": revert ONLY this change (files from the backup, flag z_bar_* removed, text back, zbar.json removed), restart,
#             and re-start the task-level cap watchdog (affine-cap-watchdog) so the cap change keeps its auto-revert.
set -u; export XDG_RUNTIME_DIR=/run/user/$(id -u) TZ=America/Costa_Rica
RA=/home/const/subnet120-rewrite/state/root-audits; B=$RA/20261101-zbar-stepdown; C=$B/code; CAPB=$RA/20261101-cap-task-level
GS=$RA/20261026-promotion-gate/state; FLAG=$RA/20261023-toploc-credit-train-gate/staged/PROMOTION-GATE.json
CODE=$RA/20261026-promotion-gate/code; WEB=/home/const/subnet120/affine/website; LE=/home/const/subnet120-rewrite/state/mimo-mainnet-v1/epochs
PY=/home/const/subnet120-rewrite/.venv/bin/python; LOGF=$B/state/activate.log
say(){ echo "$(date -u +%FT%TZ) $(date '+%T CR') $*" | tee -a $LOGF; }
prog(){ echo "- $(date '+%H:%M CR'): activate_zbar: $*" >> $B/PROGRESS.md; }
restart_all(){ systemctl --user restart affine-promotion-gate; systemctl --user restart affine-training-params affine-gate-ledger affine-k1-transparency; sleep 5; }
cap_wd(){ [[ -f $CAPB/state/REVERTED ]] && return; systemctl --user reset-failed affine-cap-watchdog 2>/dev/null
  systemd-run --user --unit=affine-cap-watchdog --description="SN120 task-level cap watchdog (auto-revert)" $PY -u $CAPB/code/watchdog_cap.py $(date +%s) $(cat $CAPB/state/FROM_EPOCH) && say "cap watchdog re-started"; }
revert(){
  WHY=$1; BK=$(cat $B/state/LAST-BACKUP); E=$(cat $B/state/FROM_EPOCH)
  [[ ${FROM_WATCHDOG:-} == 1 ]] || systemctl --user stop affine-zbar-watchdog 2>/dev/null
  cp -p $BK/baseline_v1.py $BK/prior_gate.py $BK/gate_ledger.py $CODE/ && cp -p $BK/baseline_v1.py $BK/prior_gate.py $WEB/gate/heldout/
  cmp -s $WEB/k1.js $B/state/k1.js.live && cp -p $BK/k1.js $WEB/k1.js || say "WARN: live k1.js changed since install: left as is"
  sed -i "s#/k1.js?v=[^\"]*#/k1.js?v=$(cat $B/state/K1TAG)#" $WEB/index.html
  flock -w 60 $FLAG.lock python3 $C/setflag_zbar.py $FLAG off $E "z bar step-down revert: $WHY"
  python3 $C/patch_texts_zbar.py revert $E $WEB/llms.txt || say "WARN: text revert rc!=0"
  rm -f $WEB/gate/zbar.json
  restart_all; touch $B/state/REVERTED
  say "REVERT ($WHY): z bar step-down reverted (code/site from $BK, flag z_bar off, text back); gate=$(systemctl --user is-active affine-promotion-gate)"
  prog "REVERT ($WHY)"; cap_wd
}
if [[ ${MODE:-} == revert ]]; then revert "${WHY:-manual}"; exit 0; fi
W=${AFTER_EPOCH:?set AFTER_EPOCH}; E=$((W + 1))
[[ -f $B/state/INSTALLED ]] && { say "ABORT: already installed"; exit 1; }
say "armed: waiting for DECISION e$W and a quiet gap before e$E"; prog "armed: after e$W DECISION"
quiet(){ [[ ! -f $GS/epochs/$E/DECISION.json && ! -f $LE/$E/TRAINING-RESULT.json && ! -f $GS/GATE-BUSY.json ]]; }
while :; do
  until [[ -f $GS/epochs/$W/DECISION.json ]]; do sleep 3; done
  ok=0; for i in $(seq 1 60); do quiet && { ok=1; break; }; [[ -f $LE/$E/TRAINING-RESULT.json ]] && break; sleep 3; done
  if [[ $ok == 1 ]]; then exec 9>$RA/LIFECYCLE-ACTIVATION.lock; flock -n 9 && quiet && break; exec 9>&-; fi
  say "no quiet gap after e$W DECISION (busy/TRAINING-RESULT e$E/lock); waiting for the next DECISION"; W=$E; E=$((W + 1))
done
TS=$(date -u +%Y%m%dT%H%M%SZ); BK=$B/backups/zb-$TS; mkdir -m 700 -p $BK
cp -a $CODE/baseline_v1.py $CODE/prior_gate.py $CODE/gate_ledger.py $FLAG $WEB/llms.txt $WEB/gate.json $WEB/mining.json $WEB/k1.js $WEB/index.html $BK/ \
  && echo $BK > $B/state/LAST-BACKUP && echo $E > $B/state/FROM_EPOCH || { say "ABORT backup"; exit 1; }
grep -o '/k1.js?v=[^"]*' $WEB/index.html | head -1 | sed 's#/k1.js?v=##' > $B/state/K1TAG
for f in baseline_v1.py prior_gate.py; do sed "s/513/$E/g" $C/$f > $B/state/$f.live; done; cp $C/gate_ledger.py $B/state/gate_ledger.py.live
$PY -c "import ast,sys; [ast.parse(open(p).read()) for p in sys.argv[1:]]" $B/state/*.py.live || { say "ABORT: syntax"; exit 1; }
quiet || { say "ABORT: e$E started during backup"; exit 1; }
install -p -m 644 $B/state/baseline_v1.py.live $CODE/baseline_v1.py && install -p -m 644 $B/state/prior_gate.py.live $CODE/prior_gate.py \
  && install -p -m 644 $B/state/gate_ledger.py.live $CODE/gate_ledger.py && cp -p $CODE/baseline_v1.py $CODE/prior_gate.py $WEB/gate/heldout/ \
  || { say "ABORT: copy failed"; cp -p $BK/baseline_v1.py $BK/prior_gate.py $BK/gate_ledger.py $CODE/; exit 1; }
install -m 644 $C/k1.js $WEB/k1.js && cp -p $WEB/k1.js $B/state/k1.js.live && sed -i "s#/k1.js?v=[^\"]*#/k1.js?v=zbar-$TS#" $WEB/index.html
flock -w 60 $FLAG.lock python3 $C/setflag_zbar.py $FLAG on $E "z bar step-down activation after e$W DECISION" | tee -a $LOGF
python3 $C/patch_texts_zbar.py apply $E $WEB/llms.txt | tee -a $LOGF
systemctl --user stop affine-cap-watchdog 2>/dev/null && say "cap watchdog stopped (affine-zbar-watchdog checks the cap rule too)"
T0=$(date +%s); echo $T0 > $B/state/T0; restart_all; touch $B/state/INSTALLED
say "INSTALLED after e$W DECISION (backup $BK); first decision under the z bar step-down: e$E"; prog "INSTALLED after e$W DECISION; first e$E; backup $BK"
exec 9>&-
sleep 60
A=$(systemctl --user is-active affine-promotion-gate); P=$(systemctl --user is-active affine-training-params); L=$(systemctl --user is-active affine-gate-ledger); K=$(systemctl --user is-active affine-k1-transparency)
ERR=$(python3 - $T0 <<'PY'
import json, sys
t0 = float(sys.argv[1]); n = 0; ticks = 0; last = 0
for l in open('/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate/state/gate.log', errors='replace').readlines()[-400:]:
    try: d = json.loads(l); last = d.get('at', last)
    except Exception: n += ('Traceback' in l and last >= t0 - 5); continue
    if d.get('at', 0) < t0: continue
    n += 'error' in str(d.get('event', '')); ticks += d.get('event') == 'tick'
try: g = json.load(open('/home/const/subnet120/affine/website/gate.json')); zb = (g.get('z_bar') or {}).get('current'); fresh = g['updated_at'] > t0
except Exception: zb = None; fresh = False
print('%d %d %s %s' % (n, ticks, zb, fresh))
PY
)
set -- $ERR
if [[ $A != active || $P != active || $L != active || $K != active || $1 != 0 || ( $4 == True && $3 == None ) ]]; then revert "post-install health: gate=$A tp=$P ledger=$L k1=$K errors/ticks/zbar/fresh=$ERR"; exit 1; fi
say "LIVE gate=$A tp=$P ledger=$L k1=$K errors/ticks/zbar/gatejson-fresh=$ERR"; prog "LIVE ($ERR)"
systemd-run --user --unit=affine-zbar-watchdog --description="SN120 z bar step-down watchdog (auto-revert; also checks the cap rule)" $PY -u $C/watchdog_zbar.py $T0 $E \
  && say "watchdog started (unit affine-zbar-watchdog)" || say "WARN: watchdog failed to start"
