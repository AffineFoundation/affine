Below is a refined plan for re‑implementing the Affine codebase.  It preserves Affine’s role as a Bittensor subnet but adopts a new, more principled architecture inspired by the recent Ridges incentive mechanism and other RL research.  The plan focuses on simplicity, determinism and robustness.

---

## 1. Repository overhaul

* **Delete existing modules.** Remove the current AgentGym implementation, Pareto dominance logic, round‑robin sampling and any other old mechanisms that conflict with the new design.
* **Adopt a minimalist structure.** Organize the new code into a few clear packages:

  * `affine/envs`: Gym‑style environments (Tic‑Tac‑Toe and eight‑digit multiplication).
  * `affine/evaluation`: head‑to‑head scoring logic, Wilson score interval calculations, environment aggregation and ratio‑to‑beat tracking.
  * `affine/validator`: validator sampling loops, block‑chain logging, vTrust scoring and weight setting.
  * `affine/miner`: miner utilities for model upload/query.
  * `affine/cli`: entry points (`af validate`, `af mine`, etc.) with minimal CLI glue.
* **Remove external complexity.** Do not reinstate AgentGym, Pareto frontier algorithms, multi‑agent round‑robin scheduling or any legacy scaffolding.

## 2. Gym‑based deterministic environments

### 2.1 General design

* **Use the Gym (Farama/OpenAI) interface.** Each environment is a subclass of `gym.Env` with `observation_space`, `action_space`, `reset()`, and `step()` methods.
* **Procedural generation with challenge IDs.**

  * Every call to `reset()` returns an initial observation and an `info` dictionary containing a `challenge_id` (a UUID or a deterministic hash).  The `challenge_id` encodes the seed so that all validators can regenerate the exact same task.
  * Additional metadata (difficulty, version, etc.) is included in `info` to aid external verification.
  * Procedural content generation uses only the seed from the `challenge_id` so tasks are reproducible.

### 2.2 Initial environments

1. **Tic‑Tac‑Toe (multi‑turn):**

   * Observation: 3×3 board state and player to move.
   * Action space: discrete index 0–8 for placing a mark.
   * Reward: +1 for a win, –1 for a loss, 0 for a draw.  Games end when a terminal state is reached.
   * The environment must work deterministically given the `challenge_id` (e.g., it might choose which player starts or predetermined board positions).

2. **Eight‑digit multiplication (single‑turn):**

   * Observation: two eight‑digit numbers to multiply, given as strings or vectors.
   * Action space: the agent outputs a string representing the product.
   * Reward: +1 if exactly correct, 0 otherwise.
   * The challenge generation seeds the two numbers from the `challenge_id`.

## 3. Head‑to‑head evaluation and scoring

### 3.1 Contender vs. champion

* **Champion-centric evaluation.** Maintain a single “best” (champion) miner.  When a new miner registers, it becomes the “contender.”  We only compare contender vs. champion instead of round‑robin tournaments.
* **Pairwise sampling per environment.** For each environment, generate challenges and have both miners respond.  Track which model performs better on each challenge (win/lose/draw).
* **Wilson score interval for stopping.**

  * Use the Wilson score interval to estimate the true win rate of the contender over the champion.  The Wilson interval is asymmetric and is reliable even with small sample sizes.
  * Continue sampling until the lower bound of the contender’s interval exceeds 0 (meaning the contender is better) or the upper bound falls below 0 (champion remains superior).  This gives a statistically sound decision, mitigating false positives.
* **Parallel environment evaluation with early stopping.**

  * Evaluate all environments in parallel (tic‑tac‑toe and multiplication).  Within each environment, run successive challenges until the Wilson interval yields a confident decision.
  * After each environment decision, update a global tally: if the contender has already lost enough environments that it cannot reach the overall majority, abort the remaining tests.
  * When the contender wins more than half of the total environments (i.e., `(N+1)/2` out of `N`), declare it the new champion and stop further tests.

### 3.2 Adaptive “ratio to beat”

* **Initial threshold.** A new miner must only be “better than 50 % + ε” with high confidence to dethrone the current champion.
* **Geometric mean update.** Whenever a challenger wins, compute the geometric mean of its win rates across environments and update the “ratio to beat” accordingly.  A future contender must exceed this ratio to dethrone the current champion.
* **Exponential decay.** Apply exponential decay to the ratio over time so that the threshold gradually relaxes; this prevents the frontier from stagnating if no strong challengers appear.

## 4. Independent validator sampling and shared blocks

### 4.1 Sampling and transcripts

* **Independent sampling.** Each validator independently generates challenges using the environment’s `reset()` function, queries the contender and champion (via Chutes inference endpoints), and records the responses.
* **Detailed transcripts.** For every challenge, validators store:

  * `challenge_id` and environment name.
  * A unique invocation identifier (to detect duplicate queries).
  * Full conversation transcript (prompt, model response) plus the deterministic evaluation score.
* **Block logs with hashes.** Validators group transcripts into “blocks.”  Each block contains a cryptographic hash of the block’s contents and references the hash of the previous block, forming a chain.  This ensures auditability and prevents tampering.

### 4.2 vTrust and weight setting

* **vTrust metric.** Track each validator’s accuracy in reporting results.  If a validator’s reported winner conflicts with the majority of other validators for the same challenge, their vTrust decreases.  Higher vTrust validators have more influence when weights are set.
* **Winner‑takes‑all weights.** After aggregating data from all validators, set the Bittensor weights such that the winning miner receives all the stake; others receive zero.  This is similar to Ridges’ system where validators run code on 50 SWE‑bench tasks and give all reward to the highest‑scoring agent.
* **On‑chain commitments.** Validators sign their block hashes on‑chain to prove they executed the evaluation.  Any validator can re‑run the challenges using the `challenge_id` to verify the results.

## 5. Additional considerations for simplicity and robustness

* **Deterministic evaluation only.** Do not allow randomness in agent evaluation aside from the deterministic seed from `challenge_id`.  This prevents exploits or test leakage.
* **Minimal dependencies.** Use only Gym (for environments), standard Python libraries and Bittensor/Chutes SDKs.  Avoid heavy frameworks.
* **Security and sandboxing.** When running miner models, use Chutes’ secure inference endpoints so validators never execute untrusted code locally (similar to Ridges’ inference and embedding proxies).
* **Clear CLI.** Provide a small set of commands: `validate` (runs validator loop), `mine` (submits a model, queries environment), `env test` (generates a challenge for debugging), and `ratio show` (shows current ratio to beat).
* **Documentation and guidelines.** Include thorough docstrings and usage examples, but keep code minimal and well commented.  Provide guidance on how to add new Gym environments (ensuring deterministic generation and metadata).

---

**Why this plan?**  The current Affine design uses Pareto frontiers and round‑robin sampling; this tends to be complex and may encourage sybil attacks or overfitting.  The new design simplifies the evaluation to head‑to‑head comparisons with statistical guarantees.  Ridges’ open‑source incentive system shows that winner‑takes‑all scoring on a limited set of tasks can produce robust competition.  The Wilson score interval provides a principled way to stop evaluations early while ensuring fairness.  By using Gym environments with deterministic `challenge_id`s, validators can independently reproduce tasks and verify each other’s results, preventing cheating and improving transparency.
