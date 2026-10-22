set -euo pipefail
N=~/subnet120-rewrite/state/root-audits/20261020-measured-impact-v9
R=~/subnet120-rewrite/state/root-audits/20261022-v9-window-credit-seq7
TS=$(date -u +%Y%m%dT%H%M%SZ); B=$R/backup/pre-seq7-$TS; mkdir -p $B/pod
U=~/.config/systemd/user
POD="ssh -n -o BatchMode=yes -o ConnectTimeout=20 -p <SCORER_SSH_PORT> root@<SCORER_HOST>"
# 0. no submission in flight
python3 -c "import json,sys;s=json.load(open('$HOME/subnet120-rewrite/state/live-math-launch-preparation-v1/reward-state-v1/current-assessment-v1/submission.json'))['status'];print('submission status',s);sys.exit(1 if s=='submitting' else 0)"
# 1. backup
cd $N
cp -p weights_v9.py hourly_entry_v9.py sign_v9.py test_weights_v9.py HOST-COMPOSITION.v9-live.ROOT-SIGNED.private.json LIVE-CONFIG-v9.ACTIVE.signed.json RELEASE-v9.signed.json $B/
$POD "cat /root/sn120-scorer-v9/v9agg.py" > $B/pod/v9agg.py
$POD "cat /root/sn120-scorer-v9/test_v9agg.py" > $B/pod/test_v9agg.py
(cd $B && sha256sum *.py *.json pod/* > SHA256SUMS)
echo backup $B
# 2. pod scorer fix (-inf)
scp -q -o BatchMode=yes -P <SCORER_SSH_PORT> $R/staged/v9agg.py $R/staged/test_v9agg.py root@<SCORER_HOST>:/root/sn120-scorer-v9/
$POD "cd /root/sn120-scorer-v9 && /root/sampler-venv/bin/python test_v9agg.py | tail -1"
# 3. writer code
cp $R/staged/weights_v9.py $R/staged/hourly_entry_v9.py $R/staged/sign_v9.py $R/staged/test_weights_v9.py $R/staged/public_scores_v9.py $N/
chmod 644 $N/public_scores_v9.py
cd $N && ~/subnet120-rewrite/.venv/bin/python -B test_weights_v9.py 2>&1 | tail -1
# 4. sign: live-config seq4 (window rule), host composition (new pins), release seq7
PY=~/subnet120-rewrite/.venv/bin/python
$PY -B sign_v9.py live-config --note "seq7 Jacob 2026-10-22 11:10 CR: window-credit-v1 replaces always-pay top-10 softmax + share-EMA. credit = groups*(1-p) per 3-group unit over the last 4 sealed epochs; share proportional to window credit; zero-groups floor (>=1 verified group in window and >=1 in newest scored epoch, else 0); public scores at https://affine.io/scores.json" --window-payout '{"rule":"window-credit-v1","window_epochs":4,"min_window_groups":1,"min_newest_groups":1}'
$PY -B sign_v9.py host
$PY -B sign_v9.py release --note "seq7: window-credit-v1 writer rule + zero-groups floor + public scores publisher; scorer v9agg p-value continuity fix (p=1 -> -inf broke e85/e87)"
# 5. check
$PY -I -B hourly_entry_v9.py --policy HOST-COMPOSITION.v9-live.ROOT-SIGNED.private.json --authority 3301134b38401196d006a621ac4a772bb4b0e6afa15a7d34a0d1ae5f2c630bcd --check | head -c 1500; echo
sha256sum LIVE-CONFIG-v9.ACTIVE.signed.json HOST-COMPOSITION.v9-live.ROOT-SIGNED.private.json RELEASE-v9.signed.json weights_v9.py hourly_entry_v9.py public_scores_v9.py
