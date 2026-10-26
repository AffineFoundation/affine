# Prospective training ratio guard

Pure CPU diagnostics for a new research experiment. This does not alter the
original fixed-group study, live miner proof rules, rewards or training services.

```python
from research.group_ratio_guard import Thresholds, Trajectory, assess

# The caller has authenticated complete groups and finished ALL initial
# training-mode prefills. No backward or optimizer step has occurred.
rows = [Trajectory(task_id, nonce, outcome, advantage,
                   tuple(actual_generation_lp), tuple(training_prefill_lp))]
report = assess(rows, Thresholds())
persist_original(report)
if not report['passed']:
    abort_original_without_replacing_rows()
```

Identity, complete-group validation, actual sampling witnesses, advantages and
population completeness are caller responsibilities. Every intended loss-bearing
trajectory must be included. Unresolved rows may be omitted if they were not
prefilled; if supplied, mark them inactive. Their LP arrays are still validated,
but inactive rows cannot dilute active training metrics. No likelihood is changed.

Defaults are max absolute log-ratio 1, mean .02, P99 .15, pooled PPO-outside-band
fraction .01, maximum per-trajectory fraction .05, epsilon .2. Bounds are
inclusive. These are prospective numerical usability thresholds informed by four
TRAIN groups, not proof tolerances or statistically calibrated all-task limits.
The original four groups are all-correct and inadmissible to mixed training;
their zero advantages cannot establish active gradient clipping on mixed groups.

The report distinguishes outside-band tokens from advantage-sign-aware clipped
gradients. A zero-advantage population has an undefined sign-aware rate (`None`).
Malformed, nonfinite, unaligned or duplicate-identity input raises
`RatioInputError`; numerical threshold failures return a complete failed report.
Keep the report and original input on failure rather than silently filtering.

Run CPU controls:

```sh
.venv/bin/python -B -m unittest research.group_ratio_guard.test_guard -v
```
