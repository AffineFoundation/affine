# v9 seq6 score EMA — rollback  (2026-10-21, CR = UTC-6)
Activated 19:02 CR (01:02:52Z). Change = weights_v9.py share-EMA (config-switched) + LIVE-CONFIG seq3 `smoothing` field.
Scorer, pod, trainer, lifecycle, admission, mech-1, writer base: untouched.
N=~/subnet120-rewrite/state/root-audits/20261020-measured-impact-v9
B=~/subnet120-rewrite/state/root-audits/20261021-v9-score-ema/backup/pre-seq6-ema-20261022T010241Z  (SHA256SUMS inside)

## 0. Always first
python3 -c "import json;print(json.load(open('$HOME/subnet120-rewrite/state/live-math-launch-preparation-v1/reward-state-v1/current-assessment-v1/submission.json'))['status'])"
If `submitting`, wait for that writer run to finish.

## FAST rollback (no re-sign; new code + seq2 config = exact seq5 rule, verified offline e70-e79 diff 0.0)
cd $N && mv LIVE-CONFIG-v9.seq3.signed.json LIVE-CONFIG-v9.seq3.rolled-back.signed.json && cp -p LIVE-CONFIG-v9.seq2.signed.json LIVE-CONFIG-v9.ACTIVE.signed.json
Takes effect at the next hourly window. Check: ~/subnet120-rewrite/.venv/bin/python -I -B hourly_entry_v9.py --policy HOST-COMPOSITION.v9-live.ROOT-SIGNED.private.json --authority 3301134b38401196d006a621ac4a772bb4b0e6afa15a7d34a0d1ae5f2c630bcd --check   (config_sequence 2, no `smoothing` in v9)

## FULL rollback to seq5 code
cd $N && cp -p $B/weights_v9.py $B/sign_v9.py $B/test_weights_v9.py . \
 && cp -p $B/HOST-COMPOSITION.v9-live.ROOT-SIGNED.private.json $B/LIVE-CONFIG-v9.ACTIVE.signed.json $B/RELEASE-v9.signed.json .
(all three signed docs are the originals, so pins match the restored code; scorer driver re-verifies RELEASE seq5)
Then the --check above. Leave LIVE-CONFIG-v9.seq3 / RELEASE-v9.seq6 files in place (history), or move them aside before the next sign_v9.py run (O_EXCL).

## Tuning instead of rollback
sign_v9.py live-config --payout '<same>' --smoothing '{"rule":"share-ema-v1","half_life_epochs":H,"lookback_epochs":N,"min_share":M}'  (no host/release re-sign needed)
