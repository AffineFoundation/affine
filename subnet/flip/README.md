# SN120 flip-incentive scorer (LIVE mech-0 weights)

Production copy of the scorer that produces https://affine.io/scores.json, which the
hourly mech-0 weight writer reads (`miners[].share_now`, keyed by miner public key).

| File | Role |
| --- | --- |
| `flip_scorer.py` | Joins DISPATCH-selected (miner, task) rows with the validator's probe results at consecutive checkpoints and scores flips |
| `config.json` | Live config (paths are the validator's; nothing secret) |
| `run_shadow.py` | 5-minute oneshot entry point (`affine-flip-shadow.service`) |
| `dispatch_feed.py`, `chain_uid_map.py` | DISPATCH index feed and read-only chain hotkey→UID map |
| `test_flip_symmetric.py` | Unit test for the symmetric credit mode (`python -m pytest -q test_flip_symmetric.py`) |

## Credit rule (`formula.credit_mode`)

Since 2026-10-26 (first weight window 12:00 PM CR / UTC-6, epoch 148) the live mode is
`"symmetric"`:

- per-task credit = change in estimated pass probability; with one probe sample per side,
  fail→pass = **+w**, pass→fail = **−w**, same `w = formula.symmetric_weight = 1.0`;
- a miner's `flip_score` is the **net** sum over the whole window (credit epochs ≥ 93);
  negative nets are kept and carry forward;
- shares = `max(0, flip_score) / sum`, clipped only at this final step; the hard non-zero
  floor (last good shares, else uniform over recent DISPATCH miners) and UID bans are unchanged.

Pure noise (true pass rate unchanged) nets 0 in expectation. The previous mode,
`"surprise_asymmetric"` (2026-10-23 → 2026-10-26), paid a gain `−ln p̂` and charged a loss
`−ln(1−p̂)`; at p̂ = 0.10 that is 22:1, so noise alone earned about +0.198 per task-epoch.

Revert: set `formula.credit_mode` to `"surprise_asymmetric"` and run the scorer once.

Unchanged by this mode: eligibility (`hard_fail`), the TOPLOC / sampling verify gate and its
`unknown_policy`, UID bans, nonces, training.
