#!/usr/bin/bash
# Daily IRT refit for the live irt-info draw (RA/20261102-irt-live). Low priority (nice 19, ionice idle, 4 torch threads); waits while
# the host load is high; exports the private params only if the sanity checks in export_params.py pass (else the old params stay).
# theta of the head is not refit here: the gate recomputes it at every round open from the head s own rollouts.
set -u; export TZ=America/Costa_Rica IRT_THREADS=4
RA=/home/const/subnet120-rewrite/state/root-audits; I=$RA/20261102-irt-live; W=$RA/20261102-irt-baseline/private/refit; PY=/home/const/subnet120-rewrite/.venv/bin/python
export IRT_WORK=$W; mkdir -p -m 700 $W; LOG=$I/state/refit.log
say(){ echo "$(date "+%F %T CR") $*" >> $LOG; }
for i in $(seq 1 24); do L=$(cut -d" " -f1 /proc/loadavg); python3 -c "import sys; sys.exit(0 if $L < 16 else 1)" && break; say "load $L >= 16: wait 10 min"; [[ $i == 24 ]] && { say "skip: host busy"; exit 0; }; sleep 600; done
say "refit start (load $(cut -d" " -f1 /proc/loadavg))"
cd $I/code/refit
nice -n 19 ionice -c3 $PY -B gather.py > $W/gather.log 2>&1 || { say "gather failed: $(tail -1 $W/gather.log)"; exit 1; }
nice -n 19 ionice -c3 $PY -B fit.py ab 1.0 4 chain > $W/fit.log 2>&1 || { say "fit failed: $(tail -1 $W/fit.log)"; exit 1; }
chmod 600 $W/*.npz $W/cells.tsv 2>/dev/null
$PY -B $I/code/export_params.py $W/fit_ab_1_4_chain.npz $RA/20261102-irt-baseline/private/live/irt_params.private.json "ab_1_4_chain refit $(date +%F)" >> $LOG 2>&1 \
  && say "refit exported ($(grep -m1 cells $W/gather.log))" || say "refit NOT exported (sanity check failed; previous params kept)"
