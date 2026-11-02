#!/usr/bin/bash
# SN120 z bar FIXED at 1.00 (RA/20261102-zbar-flat; Jacob 2026-11-02 08:20 CR, pt1055u): removes the time-relaxed bar
# (E513 steps, E514+ linear drop max(0.25, 1 - 0.75 h/12)); every decision from the next quiet gap on uses z > 1.00.
# Files: CODE/{baseline_v1,prior_gate,gate_ledger,training_params}.py (+ public copies), WEB/k1.js (anchored patch) + index.html
# cache tag, flag heldout.prior_v1 (bar_z 1.0, schedule removed), WEB/llms.txt.
#   (default) AFTER_EPOCH=<e>: wait for DECISION e and a quiet gap, back up, install, restart gate/ledger/training-params,
#             60 s health check (else revert), stop affine-zlin-watchdog (its checks assume the relaxed bar), start affine-zflat-watchdog.
#   MODE=revert WHY="...": back to the linear drop (code from the backup, flag keys back, k1.js/llms.txt text back), restart.
set -u; export XDG_RUNTIME_DIR=/run/user/$(id -u) TZ=America/Costa_Rica
RA=/home/const/subnet120-rewrite/state/root-audits; B=$RA/20261102-zbar-flat; C=$B/code
GS=$RA/20261026-promotion-gate/state; FLAG=$RA/20261023-toploc-credit-train-gate/staged/PROMOTION-GATE.json
CODE=$RA/20261026-promotion-gate/code; WEB=/home/const/subnet120/affine/website; LE=/home/const/subnet120-rewrite/state/mimo-mainnet-v1/epochs
PY=/home/const/subnet120-rewrite/.venv/bin/python; LOGF=$B/state/activate.log; FILES="baseline_v1.py prior_gate.py gate_ledger.py training_params.py"
say(){ echo "$(date -u +%FT%TZ) $(date '+%T CR') $*" | tee -a $LOGF; }
prog(){ echo "- $(date '+%H:%M CR'): activate_zflat: $*" >> $B/PROGRESS.md; }
restart_all(){ systemctl --user restart affine-promotion-gate; systemctl --user restart affine-training-params affine-gate-ledger; sleep 5; }
revert(){
  WHY=$1; BK=$(cat $B/state/LAST-BACKUP); E=$(cat $B/state/FROM_EPOCH)
  [[ ${FROM_WATCHDOG:-} == 1 ]] || systemctl --user stop affine-zflat-watchdog 2>/dev/null
  for f in $FILES; do cmp -s $CODE/$f $B/state/$f.live && cp -p $BK/$f $CODE/$f || say "WARN: live $f changed since install: left as is"; done
  cp -p $BK/baseline_v1.py $BK/prior_gate.py $WEB/gate/heldout/
  python3 $C/patch_k1_zflat.py revert $WEB/k1.js || say "WARN: k1.js revert rc!=0"
  sed -i "s#/k1.js?v=[^\"]*#/k1.js?v=$(cat $B/state/K1TAG)#" $WEB/index.html
  flock -w 60 $FLAG.lock python3 $C/setflag_zflat.py $FLAG off $E "z bar fixed revert (back to the linear drop): $WHY" $BK/PROMOTION-GATE.json
  python3 $C/patch_texts_zflat.py revert $E $WEB/llms.txt || say "WARN: text revert rc!=0"
  restart_all; touch $B/state/REVERTED
  say "REVERT ($WHY): z bar back to the linear drop (code from $BK, flag keys back, k1.js/llms.txt text back); gate=$(systemctl --user is-active affine-promotion-gate). affine-zlin-watchdog NOT restarted (its FROM=e514 check would flag the fixed-bar decisions)."
  prog "REVERT ($WHY)"
}
if [[ ${MODE:-} == revert ]]; then revert "${WHY:-manual}"; exit 0; fi
W=${AFTER_EPOCH:?set AFTER_EPOCH}; E=$((W + 1))
[[ -f $B/state/INSTALLED ]] && { say "ABORT: already installed"; exit 1; }
for f in $FILES; do cmp -s $CODE/$f $B/state/$f.orig || { say "ABORT: live $f differs from the snapshot this change was built on"; exit 1; }; done
say "armed: waiting for DECISION e$W and a quiet gap before e$E"; prog "armed: after e$W DECISION"
quiet(){ [[ ! -f $GS/epochs/$E/DECISION.json && ! -f $LE/$E/TRAINING-RESULT.json && ! -f $GS/GATE-BUSY.json ]]; }
while :; do
  until [[ -f $GS/epochs/$W/DECISION.json ]]; do sleep 3; done
  ok=0; for i in $(seq 1 60); do quiet && { ok=1; break; }; [[ -f $LE/$E/TRAINING-RESULT.json ]] && break; sleep 3; done
  if [[ $ok == 1 ]]; then exec 9>$RA/LIFECYCLE-ACTIVATION.lock; flock -n 9 && quiet && break; exec 9>&-; fi
  say "no quiet gap after e$W DECISION (busy/TRAINING-RESULT e$E/lock); waiting for the next DECISION"; W=$E; E=$((W + 1))
done
TS=$(date -u +%Y%m%dT%H%M%SZ); BK=$B/backups/zf-$TS; mkdir -m 700 -p $BK
cp -a $CODE/baseline_v1.py $CODE/prior_gate.py $CODE/gate_ledger.py $CODE/training_params.py $FLAG $WEB/llms.txt $WEB/gate.json $WEB/mining.json $WEB/k1.js $WEB/index.html $WEB/gate/zbar.json $BK/ \
  && echo $BK > $B/state/LAST-BACKUP && echo $E > $B/state/FROM_EPOCH || { say "ABORT backup"; exit 1; }
grep -o '/k1.js?v=[^"]*' $WEB/index.html | head -1 | sed 's#/k1.js?v=##' > $B/state/K1TAG
for f in $FILES; do sed -e "s/__ZFFROM__/$E/g" -e "s/__ZFPREV__/$W/g" $C/$f > $B/state/$f.live; done
$PY -c "import ast,sys; [ast.parse(open(p).read()) for p in sys.argv[1:]]" $B/state/*.py.live || { say "ABORT: syntax"; exit 1; }
quiet || { say "ABORT: e$E started during backup"; exit 1; }
systemctl --user stop affine-zlin-watchdog 2>/dev/null && say "affine-zlin-watchdog stopped (affine-zflat-watchdog takes over its checks)"
ok=1; for f in $FILES; do install -p -m 644 $B/state/$f.live $CODE/$f || ok=0; done
[[ $ok == 1 ]] && cp -p $CODE/baseline_v1.py $CODE/prior_gate.py $WEB/gate/heldout/ || { say "ABORT: copy failed"; for f in $FILES; do cp -p $BK/$f $CODE/; done; exit 1; }
python3 $C/patch_k1_zflat.py apply $WEB/k1.js | tee -a $LOGF && sed -i "s#/k1.js?v=[^\"]*#/k1.js?v=zflat-$TS#" $WEB/index.html
flock -w 60 $FLAG.lock python3 $C/setflag_zflat.py $FLAG on $E "z bar fixed at 1.00 after e$W DECISION" | tee -a $LOGF
python3 $C/patch_texts_zflat.py apply $E $WEB/llms.txt | tee -a $LOGF
T0=$(date +%s); echo $T0 > $B/state/T0; restart_all; touch $B/state/INSTALLED
say "INSTALLED after e$W DECISION (backup $BK); first decision with the fixed z > 1.00 bar: e$E"; prog "INSTALLED after e$W DECISION; first e$E; backup $BK"
exec 9>&-
sleep 60
A=$(systemctl --user is-active affine-promotion-gate); P=$(systemctl --user is-active affine-training-params); L=$(systemctl --user is-active affine-gate-ledger)
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
if [[ $A != active || $P != active || $L != active || $1 != 0 || ( $4 == True && $3 != 1.0 ) ]]; then revert "post-install health: gate=$A tp=$P ledger=$L errors/ticks/zbar/fresh=$ERR"; exit 1; fi
say "LIVE gate=$A tp=$P ledger=$L errors/ticks/zbar/gatejson-fresh=$ERR"; prog "LIVE ($ERR)"
systemctl --user reset-failed affine-zflat-watchdog 2>/dev/null
systemd-run --user --unit=affine-zflat-watchdog --description="SN120 z bar fixed-at-1 watchdog (auto-revert to the linear drop; also checks the cap rule)" $PY -u $C/watchdog_zflat.py $T0 $E \
  && say "watchdog started (unit affine-zflat-watchdog)" || say "WARN: watchdog failed to start"
