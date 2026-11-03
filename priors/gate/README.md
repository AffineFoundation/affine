# SN120 gate baseline prior-v1 (LIVE from 2026-11-01 17:14 CR, epoch 497 (first round r20261101T231445Z-fb048812))

Jacob 2026-11-01 14:23-14:26 CR (pt1019u): switch the held-out gate's baseline to the prior-table design. This directory is the
code of the live gate baseline since 2026-11-01 17:14 CR, epoch 497 (first round r20261101T231445Z-fb048812) (before: ivw-v2, uniform draw from the 7,496 MATH tasks, selection block,
inverse-variance weights). It was switched in one step at an epoch boundary (`activate_baseline.sh`); see https://affine.io/llms.txt
sections 5-7.

## The rule (`../baseline_v1.py`)

- One round per head. Pool: the public gate pool (affine_math + dapo, `miner/data/gate-pool-v1.jsonl`, published at the switch).
- Prior (from `priors/`, code public, table values private): per task s/n = the current head's solved/sampled counts from the
  prior sampler pod, x0.5 on every head change, unsampled tasks start at their env average; posterior Beta(s+1, n-s+1).
- Inclusion probability pi_i proportional to sqrt(E[p_i(1-p_i)]), mixed with a 10 % uniform floor, capped at 0.10, expected size N = 500.
  Poisson sampling: task i is drawn iff u_i < pi_i, u_i = sha256(version|draw|seed|id) / 2^64.
- Seed (commit-reveal): a fresh 32-byte secret, its commitment published when the round opens; after the upload deadline the
  gate fixes block B = finalized + 3; once B is finalized, seed = sha256(version|seed|secret|B|hash(B)). The secret, B, its hash,
  the seed and the drawn ids are revealed when the round closes (`verify_round.py` checks a reveal against the chain).
- The head is scored fresh on the draw once per round (12 samples), every candidate with k_cand = 8 independent samples per task.
- Delta = sum_env share_env x Hajek mean (candidate - head), shares = env size / pool size; SE = sqrt(sampling + rollout +
  sigma_run^2), sigma_run = 0: no run-noise floor in rounds opened after the 2026-11-03 change (0.003 before; the 2026-11-01 head-vs-itself test found no measurable run noise). **GREEN iff Delta > 0, z > 1, cap-hit < 0.33, >= 100 tasks scored.** The z bar is fixed at 1.00 (https://affine.io/gate.json `z_bar`); from E553 the bar is fixed at z > 1; E513-E552 used a time-relaxed bar. Cap-hit (task-level, from E507) = capped / all candidate rollouts on the scored baseline tasks: RED if >= 33% of the candidate's baseline answers hit the token cap (before E507: affine_math cap-hit < 0.20).
- No batch exclusion. Head and candidates are graded identically (same probe, grader, settings, tasks).
- Private: the prior table values and the per-task inclusion probabilities pi_i. So a reveal lets you verify the commitments, the
  block hash, the seed, the drawn ids and the GREEN turn's grades (`verify_round.py`), but not recompute Delta / z exactly.

## Files

| file | what |
|---|---|
| `../baseline_v1.py` | draw, seed, commit-reveal, estimator, public-output guard (pure functions, no I/O besides a read-only sqlite snapshot) |
| `prior_gate.py` | gate hooks: round open (table snapshot, secret), draw after the deadline, fresh head baseline, decisions, publication |
| `patch_gate.py` | exact-anchor, idempotent, reversible hooks into `gate_service.py`, `training_params.py`, `gate_ledger.py`, `k1_transparency.py`; flag `heldout.baseline = prior-v1` selects the new path, default = old path |
| `setflag_prior.py`, `activate_baseline.sh`, `watchdog_prior.py` | switch / revert / dry run; watchdog auto-reverts on a stall (> 35 min without baseline progress, draw not done 35 min after the deadline) or errors |
| `verify_round.py` | anyone: verify a closed round's reveal (secret commitment, block hash on chain, seed, draw commitment, ids in pool) |
| `build_public_pool.py` | builds `gate-pool-v1.jsonl` + manifest; drops tasks matching the private perf holdout (hash / normalised hash / 3-gram Jaccard >= 0.9) |
| `probe/` | probe side: combined snapshot (math 0..7495 + dapo at 7496+j), envs.json entry `affine_gatepool`, per-env gold in the lenient grader |
| `dry/` | offline replays: `replay_rounds.py` (real rounds, census + draws), `e2e_dry.py` + `fakepod.py` (full round on patched copies, synthetic grades) |
| `../dapo/build_dapo.py` | the pinned, decontaminated DAPO-Math-17k build |
| `patch_texts_prior.py`, `llms_sections_5_7.prior-v1.md` | llms.txt sections 5-7 for the switch |
