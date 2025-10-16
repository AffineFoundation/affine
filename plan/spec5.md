Implementation Specification for Affine Reboot
Overview and Scope

This specification describes how to re‑implement the Affine subnet on the Bittensor network. The goal is to create a minimal, verifiable head‑to‑head evaluation platform that fairly compares miners (AI models) using procedurally generated tasks and sets their Bittensor weights based on rigorous statistical evidence. The design removes legacy components such as the old AgentGym, Pareto frontier logic and round‑robin tournaments; instead it builds on principles from recent research (Ridges) and recommendations from the Bittensor validator documentation. Key goals include deterministic reproduction of every sample, simple and auditable code, robust exploit resistance and a clear winner‑takes‑all incentive mechanism.

Core design principles

Determinism and reproducibility – Every task instance (challenge) is defined by a challenge_id. Given the environment version and challenge ID, any validator can regenerate the exact task, re‑run the model’s actions and verify the result. This mirrors Bittensor’s emphasis that validators must be able to compute miners’ scores offline
docs.taostats.io
.

Minimalism – Keep the codebase lean: few files, short functions and no unnecessary dependencies. Use Python’s standard library plus Gymnasium for environments. Avoid external frameworks and network calls in environments.

Exploit resistance – Use blind procedural generation with seeded PRNGs, cryptographic hash chains for validator evidence and commit–reveal for weight publication
docs.taostats.io
. Only deterministic judging is allowed; there is no subjective scoring.

Single‑champion duels – The subnet keeps a single “champion” miner. Any newly registered miner becomes the “contender” and is evaluated head‑to‑head against the champion. There is no full round‑robin tournament, which drastically reduces attack surface and cost.

Statistical stopping rules – Use the one‑sided Wilson score interval to decide when a contender has convincingly outperformed the champion (or vice versa). This provides early stopping guarantees with a configurable confidence level
itl.nist.gov
.

1. Repository Layout

Delete the existing Affine modules (AgentGym, Pareto scoring, round‑robin sampling, outdated environments) and replace them with a small, clear package structure:

affine/
  core/
    ids.py           # Challenge ID spec and blake3 hashing
    prng.py          # Stateless PRNG (NumPy PCG64) seeded from challenge_id
    types.py         # dataclasses for Challenge, Sample, Verdict, BlockHeader etc.
    judge.py         # Deterministic judging utilities shared across envs
    wilson.py        # One‑sided Wilson confidence interval helper
  envs/
    base.py          # Gymnasium adapter: reset(seed)->obs, step(action), verify()
    tictactoe.py     # Tic‑Tac‑Toe environment (multi‑turn, miner vs miner)
    mult8.py         # 8‑digit multiplication environment (single‑turn)
    registry.py      # Registry mapping env name@version → constructor
  duel/
    arena.py         # Implements contender‑vs‑champion loop per environment
    aggregate.py     # Cross‑environment aggregation and early stopping
  validator/
    sampleplan.py    # Commit–reveal schedule for independent sampling
    bucket.py        # Append‑only block store (hash chain) + HTTP/S3 adapter
    vtrust.py        # Compute validator trust from overlap accuracy
    weights.py       # Compute winner‑takes‑all weights and submit on chain
  miner/
    client.py        # Thin HTTP client to call a miner’s chute (Chutes UID)
  net/
    bittensor.py     # Set weights; wraps Bittensor SDK
    chutes.py        # Invoke remote miner via Chutes; capture request_id
  cli.py             # Entry points: `af v2 validate`, `af v2 duel`, etc.
  pyproject.toml

Notes on structure

All modules live under affine/; no top‑level scripts except cli.py.

Do not reintroduce the old AgentGym or Pareto dominance logic. The duel engine entirely replaces round‑robin tournaments and Pareto frontiers.

The envs directory houses only deterministic, Gymnasium‑compatible environments. Future tasks can be added here following the same pattern.

The validator package encapsulates all on‑chain and network interactions: sampling, evidence storage, trust estimation and weight submission.

2. Environments
2.1 Common interface (envs/base.py)

All environments extend a minimal wrapper class that adapts Farama’s Gymnasium API. This wrapper defines a contract used by validators and miners:

class AffineEnv(gym.Env):
    """Base class for deterministic Affine environments."""
    metadata: dict = {'render_modes': [], 'name': '', 'version': ''}

    def reset(self, seed: int | None = None, options: dict | None = None)
            -> tuple[Observation, dict]:
        """Reset the environment using a seed (derived from challenge_id).
        Returns an initial observation and an info dict with at least:
            - challenge_id: hex or base58 string
            - env: name@version
            - spec_hash: blake3 hash of the environment code/parameters
            - ground_truth_commitment: hash used for verifying transcripts
        """

    def step(self, action: Any)
            -> tuple[Observation, float, bool, bool, dict]:
        """Advance the environment given an agent action.  Returns
        observation, reward, terminated, truncated, info.  The info may
        include debug data but must not leak ground truth."""

    def verify(self, response: str, info: dict) -> Verdict:
        """Pure function that checks a miner’s response against ground truth.
        It must be deterministic given (challenge_id, env version) and not
        depend on any global state or network calls."""


Key requirements:

Deterministic seeding – The wrapper uses a stateless PRNG (numpy.random.PCG64) seeded from a 64‑bit integer derived by hashing (env_id‖challenge_id‖version) using blake3. A validator can recompute the seed from challenge_id to regenerate the same task. The Procgen benchmark uses a similar pattern for reproducible procedurally generated games.

Metadata – The info dict returned by reset() must include challenge_id, env (name@version), difficulty (if applicable), spec_hash (hash of the environment code and parameters) and ground_truth_commitment (cryptographic commitment used by validators to check transcripts). Inclusion of these fields allows external parties to verify that validators have executed the task correctly.

No network calls – All randomness and ground truth are derived from the seed; the environment must not query external APIs or the internet. This prevents leakage of ground truth or injection of side channels.

2.2 Tic‑Tac‑Toe environment (tictactoe.py)

Purpose: Provide a simple multi‑turn game where miners must act sequentially. The environment pairs the contender and champion against each other on a 3×3 Tic‑Tac‑Toe board.

Key details:

Observation space: A flat 9‑cell board plus a binary indicator of which player moves next (X or O). Represent as a simple array or string ('XOX…').

Action space: Discrete(9) selecting a cell index. Invalid moves result in immediate loss.

Procedural start position: Optionally generate a partially filled board using the PRNG. In version 1.0 this can be left empty for simplicity.

Opponent logic: The side not controlled by the miner plays a perfect minimax strategy. Implement minimax in ≤80 lines; this ensures deterministic behaviour and a defined ground truth. The reward is +1 for a win, 0 for a draw and −1 for a loss. Draws do not count as wins for the Wilson interval.

Verification: The verify() method reconstructs the seed from challenge_id, rebuilds the game tree, replays the miner’s actions and checks if the reported winner is correct. This ensures transcripts are unambiguous.

2.3 8‑Digit Multiplication environment (mult8.py)

Purpose: Provide a single‑turn arithmetic task to test symbolic reasoning.

Observation: Two eight‑digit integers A and B encoded as strings or lists of digits.

Prompt (for miners): Compute A × B; return only the integer.

Action space: The miner returns a string representing the product. In a language‑model setting, the entire textual response may contain other tokens; the verifier extracts the last integer token sequence from the reply and compares it to the ground truth.

Reward: +1 if the extracted integer matches exactly A * B, otherwise 0. Both being correct or both being wrong counts as a tie and does not increment the sample count.

Verification: The verify() implementation parses the miner’s reply robustly (strip whitespace, handle separators) and performs the multiplication exactly using Python’s big‑integer semantics. All ground truth is derived from the seed; no answer is embedded in the prompt.

2.4 Adding future environments

New environments follow the same pattern: derive all randomness from the seed, output a challenge ID and metadata, implement a deterministic verify() and avoid side effects. Each environment is versioned (name@semver); changing the deterministic logic requires bumping the version and updating the spec_hash.

3. Duel Evaluation and Statistical Decision

Affine uses a king‑of‑the‑hill duel system: at any time there is a single champion miner. When a new miner registers, it becomes the contender and must beat the champion across multiple environments to take the crown.

3.1 Wilson score interval for per‑environment decision

For each environment, the validator tracks the contender’s wins (w) and total valid trials (n) against the champion. Ties (both correct or both wrong) are excluded. After each sample, compute the lower bound of the one‑sided Wilson confidence interval (confidence level 95 % by default). The NIST engineering statistics handbook gives the closed‑form formula for the Wilson interval
itl.nist.gov
:

𝐿
=
𝑝
^
+
𝑧
2
2
𝑛
−
𝑧
𝑝
^
(
1
−
𝑝
^
)
𝑛
+
𝑧
2
4
𝑛
2
1
+
𝑧
2
𝑛
,
L=
1+
n
z
2
	​

p
^
	​

+
2n
z
2
	​

−z
n
p
^
	​

(1−
p
^
	​

)
	​

+
4n
2
z
2
	​

	​

	​

,

where 
𝑝
^
=
𝑤
/
𝑛
p
^
	​

=w/n and 
𝑧
z is the standard normal quantile for the desired confidence level (e.g., 1.96 for 95 %). The algorithm stops when either:

L ≥ R_env – The contender is confidently better on this environment (a win).

1 − L ≥ R_env (equivalently, the contender’s upper bound ≤ 1 − R_env) – The champion holds the environment.

A maximum number of samples (n_cap) is reached – result is inconclusive.

The per‑environment target R_env starts at 0.5 + ε (e.g., 0.51) and is adjusted by the global ratio‑to‑beat schedule described below.

Pseudo‑code for one environment (ties are dropped):

def duel_env(contender, champion, env, R_env=0.5+eps, alpha=0.05, n_cap=2000):
    w = n = 0
    for challenge_id in generate_challenges(env):
        winner = play_one_game(contender, champion, env, challenge_id)
        if winner == 'contender':
            w += 1; n += 1
        elif winner == 'champion':
            n += 1
        # ties do not increment n
        if n >= 1:
            p = w / n
            L = wilson_lower_bound(p, n, alpha)
            U = 1 - wilson_lower_bound(1-p, n, alpha)
            if L >= R_env:
                return 'win', w, n, L, U
            if U <= 1 - R_env:
                return 'loss', w, n, L, U
        if n >= n_cap:
            return 'undecided', w, n, L, U

3.2 Multi‑environment aggregation

Let there be E environments in the registry. The validator runs the duel loop in parallel across all environments. Maintain counts W (number of environments the contender has won) and L (lost). After each environment resolves to win or loss, update these counts and check the global stopping rule:

Define R_glob as the current global ratio‑to‑beat (0.5 + ε initially). The contender must win at least ceil(R_glob * E) environments to become champion.

If W ≥ ceil(R_glob * E), stop early – the contender dethrones the champion.

If W + (E − W − L) < ceil(R_glob * E), the contender cannot possibly reach the target – stop and keep the champion.

Otherwise, continue evaluating remaining environments until all are decided or until further early stopping is possible.

This early stopping reduces cost: e.g., if the contender loses the first two environments out of three and R_glob = 0.51, they cannot reach ceil(0.51 * 3) = 2 wins; evaluation can stop immediately.

3.3 Ratio‑to‑beat schedule (difficulty ratchet)

Using a constant 0.5 threshold makes trivial dethronements too easy (e.g., a miner winning 51 % by chance). Instead, after each dethronement, update the ratio to beat:

For each environment where the contender won, compute a regularized win rate: r_e = (wins_e + 1) / (losses_e + 1).

Take the geometric mean 
𝑟
ˉ
r
ˉ
 of these r_e values across winning environments.

Map 
𝑟
ˉ
r
ˉ
 to a probability threshold:

𝑅
new
=
𝑟
ˉ
1
+
𝑟
ˉ
.
R
new
	​

=
1+
r
ˉ
r
ˉ
	​

.

If 
𝑟
ˉ
=
1
r
ˉ
=1 (equal performance), this yields 0.5; if 
𝑟
ˉ
>
1
r
ˉ
>1 (better performance), the new ratio is >0.5.

Set R_glob to 0.5 + (R_new − 0.5). Store the timestamp of this update.

Apply exponential decay: at each validator epoch, update R_glob:

𝑅
glob
(
𝑡
)
=
0.5
+
(
𝑅
peak
−
0.5
)
 
𝑒
−
(
𝑡
−
𝑡
peak
)
/
𝜏
,
R
glob
	​

(t)=0.5+(R
peak
	​

−0.5)e
−(t−t
peak
	​

)/τ
,

where R_peak is the ratio after the last dethronement and 
𝜏
τ is the half‑life in epochs (e.g., 14). Decay prevents lock‑in; over time the ratio drifts back to 0.5 so a new miner has a chance.

4. Validator Workflow and Evidence

Validators independently sample the miner pair (contender vs champion), record full transcripts of each challenge and publish these transcripts in verifiable blocks. They then cross‑validate each other’s evidence and set weights on chain using a winner‑takes‑all rule.

4.1 Sampling plan with commit–reveal

At the start of each epoch, each validator generates a secret seed s_v,t and publishes its commitment C_v,t = blake3(s_v,t). After a predetermined delay (one epoch), the validator reveals s_v,t, allowing others to derive the exact sequence of challenge_ids used for sampling. This commit–reveal schedule prevents cherry‑picking: validators cannot retroactively choose favourable tasks because they must sample according to their published plan. The commit–reveal mechanism is similar to Bittensor’s Commit Reveal 3.0 used to deter weight copying—weights are encrypted and only revealed after several epochs
docs.taostats.io
docs.taostats.io
.

Given a revealed seed, derive challenge IDs for each environment:

cid
𝑣
,
𝑡
,
𝑖
,
𝑒
=
blake3
(
𝑠
𝑣
,
𝑡
∥
epoch_anchor
𝑡
∥
𝑒
∥
𝑖
)
,
cid
v,t,i,e
	​

=blake3(s
v,t
	​

∥epoch_anchor
t
	​

∥e∥i),

where epoch_anchor_t is a public randomness beacon (e.g., a recent Bittensor or Bitcoin block hash), e is the environment name and i is a per‑env counter. Validators must sample tasks in the order determined by this function; any off‑schedule samples are ignored or penalised.

4.2 Recording samples and building blocks

For each challenge, the validator performs the following:

Generate challenge_id using the commit plan.

Call the contender and champion miners via Chutes (the inference gateway). Capture a request_id or invocation identifier if provided by Chutes.

Record a Sample object containing:

env: name@version
challenge_id: str
miner_uid: int
role: 'contender' | 'champion'
prompt: str
response: str
info: dict   # from env.reset(); includes metadata, spec_hash, etc.
verdict: Verdict  # ok/ko and reason from env.verify()
timing: {latency_ms: float, tokens: int}
request_id: Optional[str]


Append samples to an in‑memory list; when a batch size (e.g., 100 samples) is reached, construct a Block.

A BlockHeader contains:

prev_hash: str
height: int
created_at: int
validator: hotkey
epoch: int
sample_count: int
merkle_root: str  # root of sample hashes
signature: str    # validator signs header


Hash the concatenation of header and sample hashes using blake3 or SHA‑256 to produce the block hash. Store the block (header + list of sample hashes) locally and push it to a shared bucket (e.g., HTTP/S3/IPFS). Each validator maintains their own append‑only chain; the prev_hash field forms a tamper‑evident log.

4.3 Cross‑pull and validator trust (VTrust)

Validators fetch each other’s blocks from the shared bucket, recompute the hashes and verify the signatures. For each sample, they independently recompute the ground truth using verify() and compare it to the posted verdict. Based on the overlap, compute each validator’s VTrust score—the fraction of correctly reported samples weighted by prior evidence. Validators whose verdicts frequently disagree with the recomputed truth or whose blocks are invalid lose trust. The Bittensor docs describe VTRUST as a validator’s trust score showing the network’s trust in its reports
docs.learnbittensor.org
. A validator with low trust has little influence on the final weight setting.

4.4 Weight setting (winner takes all)

After aggregating all validators’ evidence (weighted by their VTrust), compute the global duel result. If the contender wins, set their weight to 1.0 and all other miners’ weights to 0.0. If the champion holds, keep weights unchanged. The commit‑reveal mechanism ensures that validators cannot copy weights: weights are committed to the chain but only revealed after several epochs
docs.taostats.io
. Validators must use the Bittensor SDK (bittensor.set_weights) or the CLI to submit weights within each tempo.

5. Data Types and Helper Functions
5.1 Dataclasses (core/types.py)

Use Python @dataclass definitions for clarity and immutability:

@dataclass(frozen=True)
class Challenge:
    env: str         # env name@version
    challenge_id: str
    meta: dict       # includes difficulty, spec_hash etc.

@dataclass(frozen=True)
class Verdict:
    ok: bool
    reason: str = ''

@dataclass(frozen=True)
class Sample:
    env: str
    challenge_id: str
    miner_uid: int
    role: str   # 'contender' or 'champion'
    prompt: str
    response: str
    info: dict
    ok: bool
    reason: str
    request_id: Optional[str] = None

@dataclass(frozen=True)
class BlockHeader:
    prev_hash: str
    height: int
    created_at: int
    validator: str
    epoch: int
    sample_count: int
    merkle_root: str
    signature: str

5.2 Wilson interval helper (core/wilson.py)

Implement a small function to compute the lower bound of the one‑sided Wilson interval. Input (wins, n, alpha) outputs a float. Use the formula from NIST
itl.nist.gov
. Keep the function ~10 lines.

5.3 Duel API (duel/arena.py and duel/aggregate.py)

arena.py implements duel_env(contender_uid, champion_uid, env, R_env, alpha) following the pseudo‑code in §3.1. It should yield the result (win, loss or undecided) and the statistics used. aggregate.py implements duel_many_envs(contender_uid, champion_uid, env_names, R_glob, ...) to coordinate multiple environments, apply early stopping and update R_glob when the contender wins.

5.4 Validator utilities (validator/sampleplan.py, bucket.py, vtrust.py, weights.py)

sampleplan.py – Manage commit–reveal seeds, derive challenge IDs, publish commitments and reveals.

bucket.py – Provide a filesystem‑like interface to read and write block files; support different back‑ends (local disk, HTTP, S3). Ensure canonical JSON encoding before hashing.

vtrust.py – Compute each validator’s trust as a Beta‑prior updated fraction of correct vs incorrect verdicts. Validators with too many invalid or off‑schedule samples get penalized.

weights.py – Collect weighted votes across validators and call net.bittensor.set_weights() to update miner weights.

6. Determinism & Verification Details

Challenge IDs: Represent as 128‑bit hex or base58 strings. Derive seeds by hashing (env_id‖challenge_id‖version) and taking the first 64 bits. Use blake3 or blake2s for hashing; these functions produce a uniformly distributed random seed.

Reproducibility: Third parties with (env, version, challenge_id, transcript) can fully regenerate the environment, replay the miner’s actions and verify the verdict using the open‑source code. This transparency is central to verifiability and trust.

Transcript format: Store transcripts as arrays of steps with fields t (turn number), role (env, minerA, minerB), content (prompt/response), obs_summary (optional summarised observation) and action_summary. Truncate or hash extremely long responses to a fixed length and include a hash footer for auditability.

No hidden state: All environment state must be reconstructible from the seed and the action sequence. Validators must not rely on hidden caches or remote services.

7. Anti‑Cheat Measures and Failure Modes

To maintain fairness, incorporate the following protections:

No answer leakage: Prompts must not include the correct answer (e.g., the mult8 environment asks only for the product, not verifying within the prompt). The ground truth stays on the validator side.

Timing and bandwidth signals: Record latency (ms) and response length (tokens) for each miner. Abnormally low latencies may indicate caching; extremely long latencies may indicate timeouts. Use these signals for additional trust heuristics (e.g., penalise responses that arrive after the time budget).

Duplicate detection: Each (env, challenge_id, miner_uid) combination should be unique within a sliding window. Duplicate submissions may indicate replay attacks; duplicates are de‑weighted when computing trust and win rates.

Block tamper detection: Invalid block signatures, broken prev_hash chains or mismatched merkle roots cause the block to be ignored and the validator’s trust to drop.

Commit–reveal enforcement: Validators must sample according to their published plan. Off‑schedule samples are discarded and counted as incorrect for VTrust purposes. Commit–reveal prevents weight copying by delaying the publication of weights
docs.taostats.io
.

Version pinning: Each environment has a semver version number. Changing the environment logic requires a version bump so that old challenge IDs cannot be reinterpreted.

8. Migration and Roll‑Out Plan

Freeze current Affine repository. Create a branch v2/cleanroom and stop development on the legacy code. All new work occurs on this branch.

Delete legacy modules (AgentGym, Pareto evaluation, round‑robin sampling, old environments). Remove any dependencies or build targets related to the old design.

Implement environments first. Create affine/envs/base.py, tictactoe.py, mult8.py and ensure determinism and verify() correctness. Add property tests to verify that the same challenge_id produces identical tasks and outcomes.

Implement the duel engine. Write core/wilson.py for Wilson confidence intervals, duel/arena.py for per‑environment duels and duel/aggregate.py for multi‑environment aggregation and ratio‑to‑beat updates. Add tests with synthetic Bernoulli streams to ensure correct stopping behaviour.

Implement validator tooling. Write the sampling plan commit–reveal logic, block construction and trust estimation. Integrate with Chutes (SN64) to query miners, capturing request_id when possible. Use Bittensor’s set_weights API to submit weights.

Build a minimal CLI. Provide subcommands: af v2 validate (run validator loop and publish blocks), af v2 duel (run a one‑off duel locally), af v2 env run (generate and inspect challenges for debugging), af v2 commit-plan and af v2 reveal-plan (publish and reveal sampling seeds), af v2 set-weights (recompute winner and set weights on chain).

Perform a testnet dry run. Run validators and miners on a test subnet without setting on‑chain weights. Observe the duel outcomes, block propagation and ratio‑to‑beat behaviour. Fine‑tune defaults (e.g., eps, n_cap, τ) based on initial results.

Enable on‑chain weights. After testing, allow validators to submit weights in commit–reveal mode. Monitor network metrics (VTrust, emissions, churn) and adjust parameters if necessary. Publish documentation and guidelines for miners and validators.

9. Testing Strategies

Environment determinism: Generate thousands of seeds and verify that env.reset(seed) returns identical observations and ground truth across multiple runs and machines.

Verifier invariants: For a recorded transcript, re‑run verify() and ensure the verdict matches exactly. Mutate transcripts deliberately to check that invalid transcripts are detected.

Wilson harness property tests: Simulate Bernoulli streams with known probabilities. Confirm that the Wilson test accepts with high probability when the true success rate ≥ target and rejects when the true rate is below the target.

End‑to‑end duel tests: Create dummy miners: one returns correct answers with probability p (e.g., 0.55), the other random. Confirm that the contender dethrones the champion when p exceeds the ratio‑to‑beat and that the ratio ratchets upward and then decays over epochs.

Block integrity tests: Tamper with sample hashes or block signatures and verify that cross‑validators detect the tampering and reduce VTrust.

10. Defaults and Configuration Parameters

Confidence level: 95 % (α = 0.05).

ε (initial slack): 0.01, so R_glob = 0.51 by default.

n_cap: 2 000 samples per environment (prevents endless evaluation).

Half‑life τ: 14 epochs (ratio‑to‑beat decays back to 0.5 roughly every two weeks).

Block size: 100 samples; adjust to trade off latency vs overhead.

Timeouts: 2 s per move in Tic‑Tac‑Toe; 10 s total for Mult8; 30 s maximum request time per sample. Timeouts count as losses.

VTrust prior: Beta(α=1, β=1) (Laplace prior) for computing trust from correct/incorrect counts.

11. Summary

This specification charts a complete overhaul of Affine. By adopting deterministic Gymnasium environments, a champion–contender duel system, rigorous Wilson‑score stopping rules, commit–reveal sampling and verifiable block logs, the new Affine ensures fairness, reproducibility and resistance to known attack vectors. Validators sample tasks independently yet share evidence via tamper‑evident chains; winners are determined collectively and published via on‑chain weights. When fully implemented, this design will produce a low‑line‑count codebase that is easy to audit and maintain while aligning with Bittensor’s validator model and the broader research trends in AI incentive design.
