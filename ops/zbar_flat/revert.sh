#!/bin/sh
# Revert the fixed z > 1 bar (RA/20261102-zbar-flat, Jacob pt1055u, live from E553) back to the time-relaxed linear drop:
# gate code (baseline_v1/prior_gate/gate_ledger/training_params) + public copies from /home/const/subnet120-rewrite/state/root-audits/20261102-zbar-flat/backups/zf-20261102T143608Z, flag keys back,
# k1.js chart (anchored reverse patch, other edits kept), index.html k1.js cache tag, llms.txt text; restarts gate/ledger/training-params.
# Pre-change copies of k1.js / index.html / llms.txt are in this directory too (reference only; the script does not overwrite whole site files).
set -e
MODE=revert WHY="${WHY:-manual revert.sh}" bash /home/const/subnet120-rewrite/state/root-audits/20261102-zbar-flat/code/activate_zflat.sh
echo "reverted z bar fixed-at-1 change, backup /home/const/subnet120-rewrite/state/root-audits/20261102-zbar-flat/backups/zf-20261102T143608Z"
