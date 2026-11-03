# Remove the king gain multiplier (Jacob t1150u / pt1150u, 2026-11-03)

"I think we should remove the multiplier it just adds complexity."

From the activation epoch on, each of the last (up to) 10 distinct GREEN UIDs gets exactly 1/N of the miner weight again, the same
as before the multiplier went live (E581, 2026-11-02). The nonzero floor and the burn to UID 0 are unchanged.

What was removed (no dead flags left behind):
- `k1_service.py` - `gain_mults()`, `GAIN_TEXT`, `gain_m` / `gain_settled` in scores.json and k1.json; formula back to
  `k1-last10-green-equal-v1`. `services/k1_service.py` is byte-identical to the pre-multiplier file.
- `training_params.py` - `training.k1.king_gain_multiplier` in mining.json / training.json.
- `priors/controller.py` + controller.json - `parent_hold` (it only existed so king/parent pairs reached 2,000 shared tasks).
- `tools/caps` - the gain-mult cap-regime step and its consumer.
- llms.txt section 8 and miner/README.md item 6 - the multiplier paragraph.
- the `affine-gain-mult` timer/service (disabled; unit files and GAIN-MULT state archived in the activation backup).

`activate_removal.sh` waits for an idle decision gap, takes both lifecycle locks, backs up, installs, restarts the publishers.
`tests/dryrun.py` / `dryrun.out`: on the live inputs the new weights are 10 x 0.1, identical to the pre-multiplier code.

Revert: `MODE=revert bash ~/subnet120-rewrite/state/root-audits/20261103-gain-mult-removal/activate_removal.sh` on the validator
host, then `git revert` this commit.
