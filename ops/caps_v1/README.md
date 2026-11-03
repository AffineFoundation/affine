# caps-v1: one canonical token cap (Jacob t1148u / pt1148u, 2026-11-03)

Why: after "12K everywhere" (E590) the prior pod kept serving at 8192 because its cap lived in a separate serve.env
that nothing checked. Now `caps.json` (repo root) is the only place the cap is set; see `tools/caps/README.md`.

Live from 2026-11-03 07:56 CR (idle boundary after E625), with identical behaviour: 12288 output / 16384 context,
overlong window 10096..12288, exactly-8192 outputs still count as capped.

- `services/training_params.py` - training.json / mining.json read `caps.py`; adds `training.caps`.
- `services/site_metrics.py` - dashboard per-epoch output cap from `caps.at_epoch()` (it showed the signed 8192 for E590+).
- `activation/watchdog_12k.py`, `activation/watchdog_sm.py` - log stamps were host UTC labelled CR; now real CR (UTC-6).
- `activation/activate_caps.sh` - `MODE=switch|revert`, waits for an idle epoch boundary, takes the lifecycle locks, backs up.
- `ops/pods/replace_pod.py` - pod roles can say `{CAPS:key}`; pod health checks fail if a pod is not at the caps.

Revert: `MODE=revert bash ~/subnet120-rewrite/state/root-audits/20261103-caps-canonical/activate_caps.sh` on the validator host,
then `git revert` this commit.
