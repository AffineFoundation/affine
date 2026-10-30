#!/bin/bash
# probe-speed / gate-owner OK 12:57 CR: gate never writes 'skip' (closed held-out round, candidate input != sampled head): writes a
# not-judged RED instead (v829i lifecycle accepts only green/red; skip -> timeout-promote = fail open). Installs right after e$WAIT_EPOCH
# DECISION. Also: early-staged delta make waits for TRAINING-RESULT (EPOCH-TIME delta_after_result=true).
# Revert: MODE=revert bash activate_skipred.sh   (delta part only: set delta_after_result false in EPOCH-TIME.json, no restart)
set -uo pipefail
D=/home/const/subnet120-rewrite/state/root-audits/20261030-probe-speed; GV=/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate
C=$GV/code/gate_service.py; BK=$C.bak-pre-skipred-20261030; NEW=$D/stage/gate_service_skipred.py; PY=/home/const/subnet120-rewrite/.venv/bin/python
prog(){ echo "- $(TZ=America/Costa_Rica date "+%H:%M CR"): activate_skipred ${MODE:-switch}: $*" >> $D/PROGRESS.md; }
ET=$GV/state/EPOCH-TIME.json
setet(){ $PY -c "import json,sys;p=sys.argv[1];d=json.load(open(p));d['delta_after_result']=sys.argv[2]=='1';json.dump(d,open(p+'.tmp','w'),indent=1);import os;os.replace(p+'.tmp',p)" $ET $1; }
if [[ ${MODE:-switch} == revert ]]; then setet 0; cp -p $BK $C && systemctl --user restart affine-promotion-gate && prog "REVERTED gate_service.py -> pre-skipred backup, gate restarted"; exit; fi
E=${WAIT_EPOCH:?}
while [[ ! -f $GV/state/epochs/$E/DECISION.json ]]; do sleep 1; done; sleep 8
[[ $(sha256sum < $C | cut -c1-16) == ${OLDSHA:?} ]] || { prog "live gate_service.py changed since staging - NOT installed"; exit 1; }
cp -p $C $BK && cp $NEW $C.skipred-tmp && $PY -m py_compile $C.skipred-tmp && mv $C.skipred-tmp $C || { prog "install failed"; exit 1; }
L0=$(wc -l < $GV/state/gate.log); systemctl --user restart affine-promotion-gate; sleep 60
A=$(systemctl --user is-active affine-promotion-gate); TB=$(tail -n +$L0 $GV/state/gate.log | grep -c -i "traceback\|epoch_error")
if [[ $A != active || $TB -ge 1 ]]; then cp -p $BK $C; systemctl --user restart affine-promotion-gate; prog "AUTO-REVERTED (gate=$A, errors=$TB)"; exit 1; fi
cp -p $ET $ET.bak-pre-delta-after-result; setet 1
prog "LIVE after e$E DECISION (+ delta_after_result on): gate_service.py $(sha256sum < $BK | cut -c1-16) -> $(sha256sum < $C | cut -c1-16) (skip -> not-judged RED); gate healthy 60 s. Revert: MODE=revert bash $D/activate_skipred.sh"
