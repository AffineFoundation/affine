#!/usr/bin/bash
# GATE NOISE FLOOR (RA/20261031-gate-noise-floor, Jacob pt900u 2026-10-31). One live change: the held-out gate's sigma_run floor
# 0.022 -> 0.0, so SE = sqrt(se_task^2 + sigma_run^2 (1 + 1/3)) uses the run noise measured each round from the 3 replicate head runs
# (sigma_run = max(floor, measured)). Bar unchanged (gain > 0, weighted_z > 1.00, cap-hit < 0.20, >= 100 kept). Config only: the gate
# reads PROMOTION-GATE.json "heldout" at every decision (no gate restart); each turn records its sigma_run, so reveals recompute exactly.
# Public text (mining.json/training.json heldout.se, gate.json rule, llms.txt section 6) via patch_texts.py + restart of the two
# read-only publishers (affine-training-params, affine-gate-ledger). Never touches the gate code, trainer, lifecycle, probe pod, batch cap.
#   WAIT_EPOCH=W : wait for gate DECISION of W, take RA/LIFECYCLE-ACTIVATION.lock (non-blocking: if another job holds it, wait for the
#                  next DECISION so this is a separate gap), back up, flag floor 0.0 (+history, effective E=W+1), patch texts, restart
#                  publishers, health check (else full revert).
#   MODE=revert  : floor back to 0.022 (+history), reverse text patches, restart the two publishers. Immediate.
set -u; export XDG_RUNTIME_DIR=/run/user/$(id -u)
RA=/home/const/subnet120-rewrite/state/root-audits; NF=$RA/20261031-gate-noise-floor
GS=$RA/20261026-promotion-gate/state; FLAG=$RA/20261023-toploc-credit-train-gate/staged/PROMOTION-GATE.json
CODE=$RA/20261026-promotion-gate/code; WEB=/home/const/subnet120/affine/website
LOGF=$NF/state/activate.log; mkdir -p $NF/state $NF/backups
say(){ echo "$(date -u +%FT%TZ) $(TZ=America/Costa_Rica date '+%T CR') $*" | tee -a $LOGF; }
prog(){ echo "- $(TZ=America/Costa_Rica date '+%H:%M CR'): activate_noise_floor: $*" >> $NF/PROGRESS.md; }
setfloor(){ flock -w 60 $FLAG.lock python3 - "$FLAG" "$1" "$2" "$3" <<'PY'
import json, os, sys, time
p, val, frm, by = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
d = json.load(open(p)); h = d['heldout']; hist = h.get('sigma_run_history') or [dict(epochs='271-', sigma_run_floor=0.022, authorized_by=['t820u'])]
last = hist[-1]; last['epochs'] = '%s%d' % (str(last['epochs']).split('-')[0] + '-', frm - 1)
hist.append(dict(epochs='%d-' % frm, sigma_run_floor=val, authorized_by=['pt900u'], set_by=by,
                 set_at=time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())))
h['sigma_run'] = val; h['sigma_run_history'] = hist; d['heldout'] = h
t = p + '.tmp.nf'; open(t, 'w').write(json.dumps(d, indent=1, sort_keys=True) + '\n'); os.chmod(t, 0o644); os.replace(t, p)
PY
}
restart2(){ systemctl --user restart affine-training-params affine-gate-ledger; sleep 5; }
revert(){
  N=$(cat $NF/state/EFFECTIVE_EPOCH 2>/dev/null || echo 0); NEXT=$(( $(ls $GS/epochs | sort -n | while read e; do [[ -f $GS/epochs/$e/DECISION.json ]] && echo $e; done | tail -1) + 1 ))
  setfloor 0.022 $NEXT "noise floor revert: $1"
  python3 $NF/patch_texts.py revert ${N:-0}; restart2
  say "REVERT ($1): floor 0.022 from E$NEXT; texts reverse-patched; tp=$(systemctl --user is-active affine-training-params) ledger=$(systemctl --user is-active affine-gate-ledger)"
  prog "REVERT ($1): floor back to 0.022 from E$NEXT"
}
if [[ ${MODE:-} == revert ]]; then revert "${WHY:-manual}"; exit 0; fi
W=${WAIT_EPOCH:?set WAIT_EPOCH}
python3 -c "import json,sys; h=json.load(open('$FLAG'))['heldout']; sys.exit(0 if h.get('enabled') is True and float(h.get('sigma_run',0.022))==0.022 else 1)" \
  || { say "ABORT: heldout not enabled or floor already changed"; exit 1; }
for f in $CODE/training_params.py $CODE/gate_ledger.py $WEB/llms.txt; do cp $f $NF/state/pre-$(basename $f); done
python3 $NF/patch_texts.py apply $((W + 1)) $NF/state/pre-training_params.py $NF/state/pre-gate_ledger.py $NF/state/pre-llms.txt >/dev/null \
  || { say "ABORT: text anchors missing (dry run)"; exit 1; }
say "staged ok; waiting for gate DECISION e$W"; prog "armed: waiting for gate DECISION e$W"
while :; do
  until [[ -f $GS/epochs/$W/DECISION.json ]]; do sleep 2; done; sleep 2
  exec 9>$RA/LIFECYCLE-ACTIVATION.lock
  if flock -n 9; then
    N1=$((W + 1)); if [[ -f $GS/epochs/$N1/DECISION.json ]]; then exec 9>&-; say "e$N1 already decided; next gap"; W=$N1; continue; fi
    break
  fi
  exec 9>&-; say "LIFECYCLE-ACTIVATION.lock held by another job after e$W; waiting for the next gap"; W=$((W + 1))
done
N=$((W + 1)); TS=$(date -u +%Y%m%dT%H%M%SZ); BK=$NF/backups/nf-$TS; mkdir -m 700 -p $BK
cp -a $FLAG $CODE/training_params.py $CODE/gate_ledger.py $WEB/llms.txt $WEB/mining.json $WEB/gate.json $BK/ && echo $BK > $NF/state/LAST-BACKUP \
  || { say "ABORT backup"; exit 1; }
echo $N > $NF/state/EFFECTIVE_EPOCH
setfloor 0.0 $N "gate noise floor activation after e$W DECISION"
python3 $NF/patch_texts.py apply $N || say "WARN: text patch rc!=0"
M0=$(stat -c %Y $WEB/mining.json); G0=$(stat -c %Y $WEB/gate.json); restart2
say "INSTALLED after e$W DECISION (backup $BK): heldout.sigma_run 0.0 from E$N; texts patched; publishers restarted"
prog "INSTALLED after e$W DECISION; floor 0.022 -> 0.0 from E$N; backup $BK"
exec 9>&-
ok=0; for i in $(seq 1 36); do sleep 5
  if [[ $(stat -c %Y $WEB/mining.json) -gt $M0 ]] && grep -q '"sigma_run_floor": 0.0' $WEB/mining.json && grep -q "from E$N no fixed floor" $WEB/gate.json 2>/dev/null; then ok=1; break; fi; done
A=$(systemctl --user is-active affine-promotion-gate); P=$(systemctl --user is-active affine-training-params); L=$(systemctl --user is-active affine-gate-ledger)
if [[ $A != active || $P != active || $L != active ]]; then revert "post-install health: gate=$A tp=$P ledger=$L"; exit 1; fi
if [[ $ok != 1 ]]; then say "WARN: publishers active but new text not yet seen in mining.json/gate.json after 180 s (gate.json may refresh on the next decision)"; fi
say "LIVE gate=$A tp=$P ledger=$L text_seen=$ok; floor 0.0 effective from E$N"; prog "LIVE from E$N (gate=$A tp=$P ledger=$L text_seen=$ok)"
