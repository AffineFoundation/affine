#!/usr/bin/env bash
# Revert green-only reveal (2026-10-31): restore the 4 service files, the deleted/modified public files, llms.txt; restart.
set -euo pipefail
D=/home/const/subnet120-rewrite/state/root-audits/20261031-green-only-reveal; TS=20261031T191213Z; B=$D/backup/$TS
C1=/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate/code; C2=/home/const/subnet120-rewrite/state/root-audits/20261027-k1-per-uid/code; W=/home/const/subnet120/affine/website
for f in gate_service.py gate_ledger.py training_params.py; do cp -p $C1/$f $C1/$f.new; cat $B/code/$f > $C1/$f.new; mv $C1/$f.new $C1/$f; done
cp -p $C2/k1_transparency.py $C2/k1_transparency.py.new; cat $B/code/k1_transparency.py > $C2/k1_transparency.py.new; mv $C2/k1_transparency.py.new $C2/k1_transparency.py
# restore every backed-up public file (deleted reveals, full round files, RED turn.json, indexes) as 0644
( cd $B/deleted && find . -type f -print0 ) | while IFS= read -r -d "" r; do mkdir -p "$W/$(dirname "$r")"; install -m 644 "$B/deleted/$r" "$W/$r"; done
[ -f $B/llms.txt ] && { cp -p $W/llms.txt $W/llms.txt.new; cat $B/llms.txt > $W/llms.txt.new; chmod 644 $W/llms.txt.new; mv $W/llms.txt.new $W/llms.txt; }
systemctl --user restart affine-promotion-gate.service affine-gate-ledger.service affine-training-params.service affine-k1-transparency.service
echo reverted
