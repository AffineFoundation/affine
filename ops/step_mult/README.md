# step_mult + optional train knobs (trainer v8.3.3l)

RA/20261102-step-mult. Jacob 2026-11-02: 10:19 CR "lets do this exactly like dou suggests and add a new hparam to the submission format",
11:19 CR "ship all of these as optional knobs that start with the defaults we have today", 11:48 CR "bigger steps let's go" (step_mult in {1, 2, 4} if the replay shows no harm; m = 4 did, so {1, 2}).

## What changed
* `affine-batch-v2` `train` accepts `step_mult` (integer: 1 or 2, default 1; 4 held back, see REPLAY.md) and 13 optional knobs (table in
  `docs/MINER_QUICKSTART.md` / `miner/README.md`). Every field defaults to the previous trainer.
* Trainer v8.3.3l = v8.3.3j + step_mult + the knobs (`release/` scripts; signed release `release-v833l-signed`). Changed files:
  `mimo_host_trainer_v83.py`, `ff_trainer.py`, `v8211_batched.py`, `v83_dp.py`. With every knob at its default, `TURN['opt']` and
  `v8211_batched.KNOBS` are `None` and no new code runs.
* duongna proposed a step multiplier m (1-8, default 1) that "scales both the lr and the per-tensor norm cap"; that is what step_mult is
  (not n_steps: one gradient, n_t := k * n_t before the global cap). Linear in k, not sqrt(k): one gradient means k "virtual steps" all point
  the same way and add linearly; sqrt(k) would move capped tensors sqrt(k) and uncapped ones k, which bends the direction.

## Bit-identity proof (`proof/`)
Real GREEN file e526 (head 3eceb890, published fp32 master), restricted to its 36 groups / 108 rollouts with prompt+output <= 3500 tokens
(the full file does not fit one B200 in the kit's padded engine). kit v8.3.3j vs kit v8.3.3l (same file) vs kit v8.3.3l with an
affine-batch-v2 copy setting every knob explicitly to its default: bf16 model files and fp32 master byte-identical
(`compare.json`: L0_vs_J and LD_vs_J all true). Control: `lr_scale_attn` 0.5 changes 177 tensors. Host-level tests
(`tests/test_v833l.py`, ALL PASS against the signed release) cover every knob path, the DP broadcast and the wrapper.

## Replay (`replay/`)
Offline replay of past GREEN turns: M_C(m) = M_H + m (M_C - M_H) from the published fp32 masters (exact, since the global cap does not
fire), scored on a fixed 1,897-task subset of the private holdout (8192-token harness), paired per task vs the head. See REPLAY.md.

## Activation / revert
`activation/activate_sm.sh` (MODE=stage|switch|revert): switch only at a round boundary (last decided turn GREEN), lifecycle stopped,
locks held; the z > 1 gate is unchanged. `activation/watchdog_sm.py` reverts automatically if a step_mult > 1 / non-default-knob turn
fails or its GREEN king loses > 3 sigma on the holdout vs its parent (or vs the pre-switch head). Manual revert:
`MODE=revert bash RA/20261102-step-mult/activation/activate_sm.sh`, plus `git revert` of this commit.
