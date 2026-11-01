## 5. The task pool, prompt, sampling and grader

- Gate pool (prior-v1, from __SWITCH__): **gate-pool-v1, 19,305 public tasks** in two envs:
  - affine_math: 7,492 MATH train problems (the 7,496-task pool minus 4 that are near-duplicates of the private
    performance holdout, see below), id math:<index in original-math7496.tasks.json>;
  - dapo: 11,813 DAPO-Math-17k problems with integer answers (BytedTsinghua-SIA/DAPO-Math-17k rev 65877096, de-duplicated and
    decontaminated against public math benchmarks by priors/dapo/build_dapo.py), id dapo:<position>.
  Both files are in the kit, byte-exact: miner/data/gate-pool-v1.jsonl (one JSON object per line: id, env, problem, answer),
  sha256 9e05d47c4672c8e066a15079577334f0432898ea06afc7045c82862345465a8e, with miner/data/gate-pool-v1.manifest.json
  (sources, sha256s, dropped ids). The old affine_math file (miner/data/original-math7496.tasks.json, sha256 77a4524a...)
  is unchanged.
- Holdout exclusion: a pool task is dropped if its problem matches a problem of the private performance holdout by
  sha256 of the raw text, sha256 of the normalised text, or token-3-gram Jaccard >= 0.9. 0 matched by hash; 4 affine_math
  tasks were dropped by the Jaccard rule (math:3322, math:6297, math:6512, math:6801: same template, different numbers).
- Gate prompt (both envs): [system: MANDATE, user: problem], with enable_thinking=True, where MANDATE = "Solve the math problem.
  Reason step by step in plain text, then end your response with the final answer in `\boxed{}`. Emit exactly one `\boxed{}`
  block and nothing after it." (Your batch prompt is a single user message, section 3.)
- Sampling: temperature 1.0, top_p 0.95, top_k 20, 8,192 output tokens, 12,288 context. It runs on vLLM 0.32.0 on 8 x B200
  (`--dtype bfloat16 --max-model-len 12288 --enable-prefix-caching --max-num-seqs 512`). Exact pins: miner/requirements-gate-vllm.txt.
  On the probe the pool is one task list: probe index = math index (0-7495), or 7496 + j for dapo:j.
- Grader (probe_worker.grade, the same for both envs, head and candidates): a capped reply (8,192 tokens, no EOS) is unresolved.
  Otherwise the final channel after exactly one `<think>...</think>` must hold exactly one balanced `\boxed{}`, which is
  checked with math-verify 0.9.0 against `\boxed{gold}`. If that check doesn't pass, the lenient fixes allan-lenient-v1
  (P1/P2 projection, E1-E7 answer equivalence) are tried. The source is in the kit (validator/probe/allan_src.py, kit/grade.py).

## 6. The promotion gate (held-out, prior-v1)

- Rounds: one round per head. A round opens when a new head is promoted (or the previous round reached 8 decided turns) and
  closes on a GREEN, a head change or 8 decided turns.
- The prior: a separate sampler pod continuously samples the CURRENT head on the whole gate pool (k = 4 per task per pass) and
  keeps, per task, s/n = solved/sampled counts (every count is multiplied by 0.5 when the head changes; a task without samples
  starts at its env's average). The prior's code is public (priors/ in the repo); its table values (s, n per task) are private.
  The gate reads a read-only snapshot of the table taken when the round opens; its sha256 is published with the round.
- The draw: inclusion probability pi_i proportional to sqrt(E[p_i(1-p_i)]) under Beta(s_i+1, n_i-s_i+1) (tasks the head
  sometimes solves get the most weight), mixed with a 10 % uniform floor (every task can be drawn), capped at 0.10, scaled
  so the expected size is N = 500; each task is drawn independently (Poisson sampling) with u_i < pi_i,
  u_i = sha256(version|draw|seed|task id). The seed is commit-reveal:
  - at open the gate publishes sha256(version|secret|<secret>) of a fresh 32-byte secret (/gate/heldout/current.json);
  - after the round's first upload deadline it fixes and publishes a block number B = finalized head + 3 (a future block);
  - once B is finalized: seed = sha256(version|seed|secret|B|hash of B), the draw is made and its size and
    draw_commitment = sha256(version|seed|sorted ids) are published. The secret is revealed when the round closes.
  So nobody (including the validator) can pick the sample after seeing the files, and miners can't predict it.
- Scoring: the head is scored FRESH on every drawn task (12 samples: 3 runs x 4); every candidate of the round runs k_cand = 8
  samples per drawn task (j = 0-3 use the same request seeds as head run 0: common random numbers). Same grader, same tasks,
  same settings for head and candidates. The turn batch is NOT excluded.
- Estimate: Delta = sum over envs of share_e x Hajek mean_e(candidate rate - head rate), where the Hajek mean weights each
  drawn task by 1/pi_i (share_e = env size / pool size: affine_math 0.388, dapo 0.612). SE = sqrt(sampling variance of the
  Hajek estimate + rollout variance + tau^2 x 2 x A^2), tau = 0.0457 (head run-to-run spread). z = Delta / SE.
- **GREEN iff Delta > 0 AND z > 1.00 (strict) AND estimated cap-hit on affine_math < 0.20 AND >= 100 drawn tasks scored
  (and >= 95 % of the draw).** With N = 500 and k_cand = 8 the bar (= SE) is expected near +1.2 points.
- GREEN: the candidate becomes the next head (bf16 + fp32 master), the turn is published and the round closes. RED: the head is kept.
- Settings and their history: https://affine.io/mining.json (`heldout`), miner/validator/gate-config.public.json. The code:
  priors/baseline_v1.py (draw, seed, estimator, rule) and priors/gate/prior_gate.py (the gate's round logic).
- Before __SWITCH__ the gate used the ivw-v2 rule (uniform draw of 1000/2000 MATH tasks, selection block 0 with 0 < c < 4,
  inverse-variance weights); its rounds and reveals stay published unchanged.

## 7. What is published, and when <!-- green-publish-20261031 --> <!-- prior-v1-publish -->

- **When a round opens:** the round's secret commitment, settings, pool sha256 and prior-table snapshot sha256
  (/gate/heldout/current.json). After the upload deadline: the block number B. After B is finalized: the draw size and
  draw_commitment. Never while the round is open: the seed, the secret, the task ids, any per-task number.
- **Each turn:** the decision with z, Delta, SE and cap rate (gate.json, turn.json).
- **When a round closes (every round, GREEN or not):** /gate/heldout/<round>.json reveals the secret, B and its hash, the seed,
  the drawn task ids and tasks (id, env, problem, answer), every turn's decision numbers, and checks that the secret, seed and
  draw match the commitments. For a GREEN-closed round it also has the GREEN turn's per-task head and candidate grades.
  RED candidates' per-task grades are never published. The inclusion probabilities pi_i (computed from the private prior table)
  are not published.
- **After each GREEN:** the winning file, byte-exact (/k1/turns/<epoch>/batch.json), its lr / kl and trainer report
  (/k1/turns/<epoch>/turn.json). gate.json lists every decision; the round file has the GREEN turn's grades.
- **RED turns:** only the decision, z / Delta, the counts and the file's sha256. The file, its lr / kl and the per-rollout
  trajectories stay private.
- A new round always uses a fresh secret and a fresh block hash, so a revealed round says nothing about the next sample.

