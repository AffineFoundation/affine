#!/usr/bin/bash
# SN120 task-level cap rule (RA/20261101-cap-task-level; Jacob 2026-11-01 19:42-19:43 CR, pt1032u): RED if >= 33% of the candidate's
# baseline answers hit the token cap (capped / all candidate rollouts on the scored tasks; replaces the affine_math-only cap < 0.20).
# Files: CODE/baseline_v1.py + prior_gate.py (and the public copies in WEB/gate/heldout), PROMOTION-GATE.json heldout.prior_v1, llms.txt.
#   (default) AFTER_EPOCH=<e>: wait for DECISION e, then for a quiet gap (epoch e+1 without TRAINING-RESULT, gate not busy), back up,
#             install, restart gate + 3 publishers, 60 s health check (else revert), start the watchdog (unit affine-cap-watchdog).
#   MODE=revert WHY="...": restore the backed-up code files, flag cap rule off (0.20 affine_math), llms.txt text back, restart.
set -u; export XDG_RUNTIME_DIR=/run/user/$(id -u) TZ=America/Costa_Rica
RA=/home/const/subnet120-rewrite/state/root-audits; B=$RA/20261101-cap-task-level; C=$B/code
GS=$RA/20261026-promotion-gate/state; FLAG=$RA/20261023-toploc-credit-train-gate/staged/PROMOTION-GATE.json
CODE=$RA/20261026-promotion-gate/code; WEB=/home/const/subnet120/affine/website; LE=/home/const/subnet120-rewrite/state/mimo-mainnet-v1/epochs
PY=/home/const/subnet120-rewrite/.venv/bin/python; LOGF=$B/state/activate.log
say(){ echo "$(date -u +%FT%TZ) $(date '+%T CR') $*" | tee -a $LOGF; }
prog(){ echo "- $(date '+%H:%M CR'): activate_cap: $*" >> $B/PROGRESS.md; }
restart_all(){ systemctl --user restart affine-promotion-gate; systemctl --user restart affine-training-params affine-gate-ledger affine-k1-transparency; sleep 5; }
revert(){
  WHY=$1; BK=$(cat $B/state/LAST-BACKUP)
  [[ ${FROM_WATCHDOG:-} == 1 ]] || systemctl --user stop affine-cap-watchdog 2>/dev/null
  cp -p $BK/baseline_v1.py $BK/prior_gate.py $CODE/ && cp -p $BK/baseline_v1.py $BK/prior_gate.py $WEB/gate/heldout/
  flock -w 60 $FLAG.lock python3 $C/setflag_cap.py $FLAG off "$(cat $B/state/FROM_EPOCH)" "task-level cap revert: $WHY"
  python3 $C/patch_texts_cap.py revert "$(cat $B/state/FROM_EPOCH)" $WEB/llms.txt || say "WARN: text revert rc!=0"
  restart_all
  say "REVERT ($WHY): code restored from $BK, flag cap rule off, text back; gate=$(systemctl --user is-active affine-promotion-gate)"
  prog "REVERT ($WHY)"; touch $B/state/REVERTED
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
TS=$(date -u +%Y%m%dT%H%M%SZ); BK=$B/backups/cap-$TS; mkdir -m 700 -p $BK
cp -a $CODE/baseline_v1.py $CODE/prior_gate.py $FLAG $WEB/llms.txt $WEB/gate.json $WEB/mining.json $BK/ && echo $BK > $B/state/LAST-BACKUP && echo $E > $B/state/FROM_EPOCH || { say "ABORT backup"; exit 1; }
for f in baseline_v1.py prior_gate.py; do sed "s/507/$E/g" $C/$f > $B/state/$f.live && cmp -s $BK/$f $CODE/$f || { say "ABORT: live $f changed during backup"; exit 1; }; done
$PY -c "import ast,sys; [ast.parse(open(p).read()) for p in sys.argv[1:]]" $B/state/baseline_v1.py.live $B/state/prior_gate.py.live || { say "ABORT: syntax"; exit 1; }
[[ -f $LE/$E/TRAINING-RESULT.json || -f $GS/GATE-BUSY.json ]] && { say "ABORT: e$E started during backup"; exit 1; }
install -p -m 644 $B/state/baseline_v1.py.live $CODE/baseline_v1.py && install -p -m 644 $B/state/prior_gate.py.live $CODE/prior_gate.py \
  && cp -p $CODE/baseline_v1.py $CODE/prior_gate.py $WEB/gate/heldout/ || { say "ABORT: copy failed"; cp -p $BK/baseline_v1.py $BK/prior_gate.py $CODE/; exit 1; }
flock -w 60 $FLAG.lock python3 $C/setflag_cap.py $FLAG on $E "task-level cap activation after e$W DECISION" | tee -a $LOGF
python3 $C/patch_texts_cap.py apply $E $WEB/llms.txt | tee -a $LOGF
T0=$(date +%s); echo $T0 > $B/state/T0; restart_all; touch $B/state/INSTALLED
say "INSTALLED after e$W DECISION (backup $BK); first decision under the task-level cap: e$E"; prog "INSTALLED after e$W DECISION; first e$E; backup $BK"
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
print('%d %d' % (n, ticks))
PY
)
if [[ $A != active || $P != active || $L != active || $K != active || ${ERR%% *} != 0 ]]; then revert "post-install health: gate=$A tp=$P ledger=$L k1=$K errors/ticks=$ERR"; exit 1; fi
say "LIVE gate=$A tp=$P ledger=$L k1=$K errors/ticks=$ERR"; prog "LIVE (errors/ticks=$ERR)"
systemd-run --user --unit=affine-cap-watchdog --description="SN120 task-level cap watchdog (auto-revert)" $PY -u $C/watchdog_cap.py $T0 $E \
  && say "watchdog started (unit affine-cap-watchdog)" || say "WARN: watchdog failed to start"
