#!/usr/bin/bash
# SN120 irt-info baseline draw: LIVE SWITCH (RA/20261102-irt-live; Jacob 2026-11-02 10:23 CR "ship this right now", pt1083u).
# Installs baseline_v1.py + prior_gate.py with the irt-info weighting (inert unless heldout.prior_v1.weighting = 'irt-info'), sets the
# flag, patches llms.txt, restarts gate + 3 publishers. The open round keeps its prior-v1 draw: each round freezes its settings at open,
# so the first irt-info round is the next round that opens (new head or 8 decided turns). 60 s health check else auto-revert, then the
# watchdog unit affine-irt-watchdog.
#   (default)    install in a gap (no candidate probe / baseline in flight), taking RA/LIFECYCLE-ACTIVATION.lock
#   MODE=revert  WHY="..." : flag off, close an open irt-info round (a prior-v1 round re-opens at the next tick), code + llms.txt +
#                baseline page back, restart. One command:  bash ~/subnet120-rewrite/state/root-audits/20261102-irt-live/code/activate_irt.sh  with MODE=revert
set -u; export XDG_RUNTIME_DIR=/run/user/$(id -u) TZ=America/Costa_Rica
RA=/home/const/subnet120-rewrite/state/root-audits; I=$RA/20261102-irt-live; C=$I/code
GS=$RA/20261026-promotion-gate/state; FLAG=$RA/20261023-toploc-credit-train-gate/staged/PROMOTION-GATE.json
CODE=$RA/20261026-promotion-gate/code; WEB=/home/const/subnet120/affine/website; PY=/home/const/subnet120-rewrite/.venv/bin/python
LOGF=$I/state/activate.log; mkdir -p $I/state $I/backups; chmod 700 $I/backups
say(){ echo "$(date -u +%FT%TZ) $(date '+%T CR') $*" | tee -a $LOGF; }
prog(){ echo "- $(date '+%H:%M CR'): activate_irt: $*" >> $I/PROGRESS.md; }
restart_all(){ systemctl --user restart affine-promotion-gate; systemctl --user restart affine-training-params affine-gate-ledger affine-k1-transparency; sleep 5; }
close_irt_round(){ $PY - "$1" <<'PY'
import json, os, sys, time
from pathlib import Path
os.environ['TZ'] = 'America/Costa_Rica'; time.tzset()
H = Path('/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate/state/heldout')
try: cur = json.load(open(H / 'CURRENT.private.json'))
except Exception: print('no open round'); raise SystemExit
r = json.load(open(H / cur['id'] / 'ROUND.private.json'))
if r.get('weighting') != 'irt-info': print('open round %s is a %s round: kept' % (cur['id'], r.get('weighting') or 'prior-v1')); raise SystemExit
p = H / cur['id'] / 'CLOSED.json'
if not p.exists():
    t = p.with_name('.CLOSED.json.tmp'); t.write_text(json.dumps(dict(round_id=cur['id'], closed_at=time.time(), closed_at_cr=time.strftime('%Y-%m-%d %H:%M:%S CR (UTC-6)'),
        reason='irt-info revert: %s' % sys.argv[1], closing_epoch=None), indent=1, sort_keys=True) + '\n'); os.chmod(t, 0o644); os.replace(t, p)
(H / 'CURRENT.private.json').unlink(missing_ok=True)
print('closed irt-info round %s' % cur['id'])
PY
}
restore_code(){   # restore a file only if it is still exactly what we installed (never clobber someone else's later edit)
  local BK=$1 f
  for f in baseline_v1.py prior_gate.py; do
    for dst in $CODE/$f $WEB/gate/heldout/$f; do
      if cmp -s $dst $C/$f; then cp -p $BK/$f $dst; say "restored $dst"
      else say "WARN: $dst differs from the installed irt version: left as is (inert with the flag off)"; fi
    done
  done
}
revert(){
  WHY=$1
  [[ ${FROM_WATCHDOG:-} == 1 ]] || systemctl --user stop affine-irt-watchdog 2>/dev/null
  flock -w 60 $FLAG.lock python3 $C/setflag_irt.py $FLAG off "irt revert: $WHY"
  systemctl --user stop affine-promotion-gate; say "$(close_irt_round "$WHY")"
  BK=$(cat $I/state/LAST-BACKUP 2>/dev/null); [[ -n $BK ]] && restore_code $BK
  python3 $C/patch_texts_irt.py revert $WEB/llms.txt || say "WARN llms revert"
  [[ -x $I/code/site/revert-site.sh ]] && bash $I/code/site/revert-site.sh || true
  restart_all
  say "REVERT ($WHY): flag weighting prior-v1, irt round closed, code/texts back; gate=$(systemctl --user is-active affine-promotion-gate)"
  prog "REVERT ($WHY)"; touch $I/state/REVERTED
}
if [[ ${MODE:-} == revert ]]; then revert "${WHY:-manual}"; exit 0; fi
# ---------------- install
[[ -f $I/state/REVERTED ]] && { say "ABORT: REVERTED marker present (remove it to re-arm)"; exit 1; }
python3 -c "import json,sys; h=json.load(open('$FLAG'))['heldout']; sys.exit(0 if h.get('baseline') == 'prior-v1' else 1)" || { say "ABORT: baseline is not prior-v1"; exit 1; }
for f in baseline_v1.py prior_gate.py; do cmp -s $CODE/$f $C/orig/$f || { say "ABORT: live $f changed since the irt patch was built (rebuild it)"; exit 1; }; done
[[ -s $RA/20261102-irt-baseline/private/live/irt_params.private.json ]] || { say "ABORT: irt params missing"; exit 1; }
(cd $C && nice -n 10 $PY -B test_irt.py > $I/state/preflight.txt 2>&1) && grep -q 'bitwise identical: True' $I/state/preflight.txt || { say "ABORT: preflight failed"; exit 1; }
say "preflight ok: $(grep -E 'theta|expected n|TV' $I/state/preflight.txt | tr '\n' ' ' | cut -c1-400)"
gap(){ $PY - <<'PY'
import json, sys
L = open('/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate/state/gate.log', errors='replace').readlines()[-50:]
ok = {'no-sample', 'decided', 'no-candidate', 'restored', None, True, 'none'}
for l in reversed(L):
    try: d = json.loads(l)
    except Exception: continue
    if d.get('event') == 'tick':
        st = d.get('state') or {}; bad = {k: v for k, v in st.items() if v not in ok and not isinstance(v, bool)}
        print('tick', d.get('at_cr'), 'busy' if bad else 'idle', bad); sys.exit(1 if bad else 0)
sys.exit(1)
PY
}
for i in $(seq 1 90); do G=$(gap) && break; [[ $i == 90 ]] && { say "ABORT: no idle gap in 45 min ($G)"; exit 1; }; sleep 30; done
say "gap: $G"
exec 9>$RA/LIFECYCLE-ACTIVATION.lock; flock -w 120 9 || { say "ABORT: LIFECYCLE-ACTIVATION.lock busy"; exit 1; }
TS=$(date -u +%Y%m%dT%H%M%SZ); BK=$I/backups/irt-$TS; mkdir -m 700 -p $BK; T="$(date '+%Y-%m-%d %H:%M') CR"
cp -a $FLAG $CODE/baseline_v1.py $CODE/prior_gate.py $WEB/llms.txt $WEB/mining.json $BK/ && echo $BK > $I/state/LAST-BACKUP || { say "ABORT backup"; exit 1; }
echo "$T" > $I/state/SWITCH_CR
cp $C/baseline_v1.py $C/prior_gate.py $CODE/ && cp $C/baseline_v1.py $C/prior_gate.py $WEB/gate/heldout/ || { say "ABORT: code copy failed"; cp -p $BK/baseline_v1.py $BK/prior_gate.py $CODE/; exit 1; }
chmod 644 $CODE/baseline_v1.py $CODE/prior_gate.py $WEB/gate/heldout/baseline_v1.py $WEB/gate/heldout/prior_gate.py
flock -w 60 $FLAG.lock python3 $C/setflag_irt.py $FLAG on "irt-info activation $T"
python3 $C/patch_texts_irt.py apply $WEB/llms.txt "$T" || say "WARN: llms patch rc!=0"
T0=$(date +%s); restart_all
say "INSTALLED (backup $BK): code, flag weighting=irt-info, llms.txt; services restarted"; prog "INSTALLED; backup $BK; switch $T"
exec 9>&-
sleep 60
A=$(systemctl --user is-active affine-promotion-gate); P=$(systemctl --user is-active affine-training-params); L=$(systemctl --user is-active affine-gate-ledger); K=$(systemctl --user is-active affine-k1-transparency)
ERR=$(python3 - $T0 <<'PY'
import json, sys
t0 = float(sys.argv[1]); n = 0; ticks = 0
for l in open('/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate/state/gate.log', errors='replace').readlines()[-400:]:
    try: d = json.loads(l)
    except Exception: n += 'Traceback' in l; continue
    if d.get('at', 0) < t0: continue
    n += 'error' in d.get('event', ''); ticks += d.get('event') == 'tick'
print('%d %d' % (n, ticks))
PY
)
if [[ $A != active || $P != active || $L != active || $K != active || ${ERR%% *} != 0 || ${ERR##* } == 0 ]]; then revert "post-install health: gate=$A tp=$P ledger=$L k1=$K errors/ticks=$ERR"; exit 1; fi
say "LIVE (armed for the next round open) gate=$A tp=$P ledger=$L k1=$K errors/ticks=$ERR"; prog "LIVE armed (errors/ticks=$ERR)"
systemd-run --user --unit=affine-irt-watchdog --description="SN120 irt-info draw watchdog (auto-revert to prior-v1 weighting)" $PY -u $C/watchdog_irt.py $T0 \
  && say "watchdog started (unit affine-irt-watchdog)" || say "WARN: watchdog failed to start"
