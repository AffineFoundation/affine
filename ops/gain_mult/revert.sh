#!/bin/bash
# Revert the king gain multiplier (RA/20261102-gain-mult): every payee back to the equal 1/N split, no parent hold.
set -u
G=/home/const/subnet120-rewrite/state/root-audits/20261102-gain-mult
B=$G/backups/pre-20261102T224102Z
RA=/home/const/subnet120-rewrite/state/root-audits
W=/home/const/subnet120/affine/website
systemctl --user disable --now affine-gain-mult.timer 2>/dev/null
python3 - <<PY
import json; p="$G/code/config.json"; c=json.load(open(p)); c["enabled"]=False; json.dump(c,open(p,"w"),indent=1)
PY
cp -p $B/k1_service.py $RA/20261027-k1-per-uid/code/k1_service.py
cp -p $B/training_params.py $RA/20261026-promotion-gate/code/training_params.py
cp -p $B/controller.py /home/const/subnet120-rewrite/priors/controller.py
cp -p $B/controller.json $RA/20261101-prior-table/controller.json
cp -p $B/llms.txt $W/llms.txt
systemctl --user restart affine-k1.service affine-training-params.service affine-prior-table.service
echo "- $(TZ=America/Costa_Rica date +%H:%M) CR: REVERTED (revert.sh): k1_service/training_params/controller/controller.json/llms.txt restored from $B; timer disabled; config enabled=false" >> $G/PROGRESS.md
echo reverted; echo "then on GitHub: git revert the gain-mult commit"
