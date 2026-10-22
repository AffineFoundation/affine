# SN120 measured-impact v9 — rollback  (2026-10-20, CR = UTC-6)

## 0. LIVE weight path (v9-only, burn to UID 0) — authorized by Jacob 12:06 CR "FULL REPLACEMENT"
Live = writer drop-in  ~/.config/systemd/user/affine-live-reward-writer.service.d/zzzz~~~~~~~~~~v9-live-weights.conf
     -> hourly_entry_v9.py + HOST-COMPOSITION.v9-live.ROOT-SIGNED.private.json (base: exact signed v7d of 20261020-writer-time-budget-v2)
     -> weights_v9.py + LIVE-CONFIG-v9.ACTIVE.signed.json ; assessments in reward-state-v1/current-assessment-v9-live-1/
Monitor = ~/.config/systemd/user/affine-weight-freshness-monitor.service.d/zz-v9-freshness.conf -> check_weight_freshness_v9.py

ROLLBACK (back to v7e = audit-based weights + v7c shadow blend 0.0, exactly as before):
  0. Check no submission is in flight (cursor must NOT be "submitting"; a v9 in-flight journal carries a UID 0 weight that the
     old journal code refuses to reconcile):
       python3 -c "import json;print(json.load(open(\"$HOME/subnet120-rewrite/state/live-math-launch-preparation-v1/reward-state-v1/current-assessment-v1/submission.json\"))[\"status\"])"
     If it says submitting, wait for the v9 writer run to finish (it reconciles itself) before rolling back.
  1. rm ~/.config/systemd/user/affine-live-reward-writer.service.d/zzzz~~~~~~~~~~v9-live-weights.conf
  2. rm ~/.config/systemd/user/affine-weight-freshness-monitor.service.d/zz-v9-freshness.conf
  3. systemctl --user daemon-reload      (the v7e drop-in zzzz~~~~~~~~~mimo-v822-v7e-budget2.conf applies again; nothing else changed)
  4. Next hourly run uses the old audit evidence. Note: the old path never pays/self-weights UID 0, so the on-chain UID 0 weight
     disappears at the first successful old-path submission.
Full pre-switch backup: backup/writer-before-v9-live-20261020T182157Z/  (all writer + monitor unit files and drop-ins (tgz),
  effective units, weights.json + submission cursor snapshots, v7b/v7c/v7e composition dirs (tgz), SHA256SUMS).
Kill switch without rollback (still v9 code, 100% burn): sign a LIVE-CONFIG with miner_share 0.0 (sign_v9.py live-config, then edit) —
  or simply let the score go stale: missing/stale/alarm => 100% UID 0.
Old-mechanism services/machines were NOT stopped or torn down (listed in REPORT); the audit evidence step is simply not called.

## 1. Stop the v9 scorer
    systemctl --user disable --now sn120-impact-scorer-v9.timer   (with the live path on, no new scores => after 2 epochs 100% burn)
    pod (optional): pkill -f score_v9.py; code+data in /root/sn120-scorer-v9

## 2. Re-enable the v7c scorer driver (paused ~11:47 CR to free GPU 5; it was failing e68-e70 because pod /dev/shm was full)
    systemctl --user start sn120-impact-scorer.timer        (stopped, not disabled)

## 3. Remove the 3-epoch embargo on public evidence
    rm ~/.config/systemd/user/affine-finalized-evidence-publication.service.d/zzzz~~v9-embargo.conf && systemctl --user daemon-reload
    (backups: backup/publisher-dropins-before-v9/, backup/publisher-effective-unit-before-v9.txt; original source untouched)

## 4. Pod /dev/shm housekeeping (reversible)
    /dev/shm/sn120-scorer/e62, e64 were MOVED to /root/shm-archive-20261020/sn120-scorer/ on the pod.
    Restore: rsync -a /root/shm-archive-20261020/sn120-scorer/ /dev/shm/sn120-scorer/

## Signing (authority 3301134b…, same key flow as v7c sign_impact.py; seed never printed)
    sign_v9.py release | live-config | host   — RELEASE-v9.signed.json pins every v9 file (Arbos + pod); re-sign after any change.

## 5. v9.1 native-label fetch (2026-10-20 ~14:35 CR, scorer-only; writer untouched)
    cp backup/fetch_epoch.pre-v91-claimed-labels.py fetch_epoch.py && ~/subnet120-rewrite/.venv/bin/python -B sign_v9.py release --note "revert v9.1"

## 6. seq5 always-pay payout (2026-10-20 16:33 CR, writer/payout only; scorer, pod, trainer, admission, mech-1 untouched)
Rule: LIVE-CONFIG seq2 carries payout={rule: always-pay-softmax-topk-v1, temperature 1.0, top_k 10, min_units 3, fallback_top_n 3}.
  eligible = window_gate miners with B >= 3 and finite stouffer_z; pool = top 10 by z among z>0 (none positive -> top 3 by z);
  share = softmax(z/T) over pool (100% of mech-0 to miners, no UID 0 burn); v9 gate, placebo and flag-rate alarms = diagnostics only;
  missing/stale/bad-signature score or empty eligible set -> 100% UID 0 as before. Unregistered pool members dropped + renormalised.
Changed files: weights_v9.py, sign_v9.py (--payout), test_weights_v9.py; re-signed HOST-COMPOSITION (pins new weights_v9.py),
  LIVE-CONFIG-v9.seq2 (ACTIVE), RELEASE-v9.seq5 (ACTIVE). Backup: backup/pre-seq5-alwayspay-20261020T223158Z/ (SHA256SUMS).
FAST ROLLBACK to the hard gate (no re-sign, new code reads legacy config = exact old rule):
  0. check submission.json status != submitting (see section 0)
  1. cp -p LIVE-CONFIG-v9.seq1.signed.json LIVE-CONFIG-v9.ACTIVE.signed.json
  (next sign_v9.py live-config would then try seq2 again and fail O_EXCL: move LIVE-CONFIG-v9.seq2.signed.json aside first)
FULL ROLLBACK to seq4 code:
  0. same check;  B=backup/pre-seq5-alwayspay-20261020T223158Z
  1. cp -p $B/weights_v9.py $B/sign_v9.py $B/test_weights_v9.py .
  2. cp -p $B/HOST-COMPOSITION.v9-live.ROOT-SIGNED.private.json $B/LIVE-CONFIG-v9.ACTIVE.signed.json $B/RELEASE-v9.signed.json .
  3. ~/subnet120-rewrite/.venv/bin/python -I -B hourly_entry_v9.py --policy HOST-COMPOSITION.v9-live.ROOT-SIGNED.private.json --authority 3301134b38401196d006a621ac4a772bb4b0e6afa15a7d34a0d1ae5f2c630bcd --check

## 7. seq6 share-EMA smoothing (2026-10-21 19:02 CR) — see ../20261021-v9-score-ema/ROLLBACK.md

## 8. seq7 window-credit + zero-groups floor + public scores (2026-10-22 11:23 CR) — see ../20261022-v9-window-credit-seq7/ROLLBACK.md
