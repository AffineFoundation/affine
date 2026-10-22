# SN120 v9 seq7 window-credit + zero-groups floor + public scores — ROLLBACK  (2026-10-22, CR = UTC-6)

Authorized by Jacob 2026-10-22 11:10 CR ("give the miners what they want ... audit the scores ... make the score more stable").
Activated ~11:23 CR (17:23Z). First hourly window using it: 12:00 CR.
N=~/subnet120-rewrite/state/root-audits/20261020-measured-impact-v9
R=~/subnet120-rewrite/state/root-audits/20261022-v9-window-credit-seq7
B=$R/backup/pre-seq7-20261022T172302Z   (SHA256SUMS inside; pod/ = original pod v9agg.py + test_v9agg.py)

## What changed
- weights_v9.py: new config-switched rule `window-credit-v1` (old seq5/seq6 paths unchanged and still selectable by config).
  credit(unit) = groups*(1-p); credit_i = sum over last 4 sealed epochs (newest scored epoch E back to E-3; missing scores skipped);
  share_i proportional to credit_i; ZERO-GROUPS FLOOR: >=1 verified group in window AND >=1 in epoch E, else 0. No top-K, no EMA.
- LIVE-CONFIG-v9.seq4 (ACTIVE) sha256 28bb2380e61690194ea7665e768786d641d5e4c00dcdbd4c681c6d8fa71b05d5:
  window_payout={rule: window-credit-v1, window_epochs 4, min_window_groups 1, min_newest_groups 1} (no payout/smoothing).
- HOST-COMPOSITION re-signed (pins new weights_v9.py / hourly_entry_v9.py) sha256 d93f9c496e63a9aedcaefd4180de63eb5a9053a72a3f0479702804d81daaf816.
- RELEASE-v9.seq7 (ACTIVE) sha256 b7e298ae5dccf499da60f45c845c0ed871e5d5b7ec92141fd297cd90c2f1a3d5 (pins public_scores_v9.py + new pod v9agg.py).
- hourly_entry_v9.py: after every run (finally), best-effort publish of https://affine.io/scores.json + /scores/e<E>.json + /scores/index.json
  via public_scores_v9.py (sha256 pinned inside the entry file; errors are printed, never raised; weights unaffected).
- pod /root/sn120-scorer-v9/v9agg.py: Null.p symmetric half-count clip so p<1 (p=1 gave Stouffer -inf -> JSON error -> e85/e87 never scored).
  Identical output for every unit with p<1 (all of e69-e88).
Untouched: scorer key, seed rule, trainer, lifecycle, admission, R2, mech-1 (Tobias), hyperparameters, writer base/journal/rate limits.

## 0. Always first
python3 -c "import json;print(json.load(open('$HOME/subnet120-rewrite/state/live-math-launch-preparation-v1/reward-state-v1/current-assessment-v1/submission.json'))['status'])"
If `submitting`, wait for that writer run to finish.

## FAST rollback to seq6 rule (no re-sign; new code reads the seq3 config = exact seq6 rule, verified 0.0 diff on the 11:00 CR vector)
cd $N && mv LIVE-CONFIG-v9.seq4.signed.json LIVE-CONFIG-v9.seq4.rolled-back.signed.json && cp -p LIVE-CONFIG-v9.seq3.signed.json LIVE-CONFIG-v9.ACTIVE.signed.json
Check: ~/subnet120-rewrite/.venv/bin/python -I -B hourly_entry_v9.py --policy HOST-COMPOSITION.v9-live.ROOT-SIGNED.private.json --authority 3301134b38401196d006a621ac4a772bb4b0e6afa15a7d34a0d1ae5f2c630bcd --check
(config_sequence 3, rule always-pay-softmax-topk-v1 + smoothing). The public scores file keeps refreshing (it then shows the v9 gate only).

## FULL rollback to seq6 code (writer + pod)
cd $N && cp -p $B/weights_v9.py $B/hourly_entry_v9.py $B/sign_v9.py $B/test_weights_v9.py . \
 && cp -p $B/HOST-COMPOSITION.v9-live.ROOT-SIGNED.private.json $B/LIVE-CONFIG-v9.ACTIVE.signed.json $B/RELEASE-v9.signed.json .
scp -P 40298 $B/pod/v9agg.py $B/pod/test_v9agg.py root@159.13.40.8:/root/sn120-scorer-v9/
(the three signed docs are the seq6 originals, so pins match the restored code; RELEASE seq6 pins the original pod files)
Then the --check above. Optionally rm public_scores_v9.py and ~/subnet120/affine/website/scores.json ~/subnet120/affine/website/scores/
Leave LIVE-CONFIG-v9.seq4 / RELEASE-v9.seq7 in place as history (move aside before the next sign_v9.py run: O_EXCL).

## Tuning instead of rollback (no host/release re-sign needed)
cd $N && ~/subnet120-rewrite/.venv/bin/python -B sign_v9.py live-config --note "..." --window-payout '{"rule":"window-credit-v1","window_epochs":W,"min_window_groups":G,"min_newest_groups":M}'

## Caution (found during this release)
hourly_entry_v9.py --plan calls the live recipients() step, which rewrites live-log/weights-<current cutoff>.json with the PLANNED
vector. The 11:24 CR plan overwrote weights-1792688400.json (the real 11:00 CR submission log); it was restored byte-identical
(sha256 2a353fa7...) from a copy taken at 11:12 CR; the plan version is kept here as weights-1792688400.overwritten-by-plan-1724Z.json.
Chain state, assessments and the journal were not affected. Do not run --plan without moving that file aside first.
