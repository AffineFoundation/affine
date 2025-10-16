# Affine reboot: a minimal, verifiable, winner-takes-all subnet

Below is a concrete, low-line-count plan to rebuild **Affine** around three pillars: (1) Gymnasium-style, procedurally generated, **verifiable** environments; (2) **duel-only** evaluation with sequential confidence (Wilson) and early stopping across multiple envs; (3) **independent validator sampling** with signed, hash-chained sample blocks and VTrust-weighted, winner-takes-all weights. It keeps Affine on **Bittensor** (SN64 miners live on **Chutes**) and aligns with Gymnasium’s API and best practices for deterministic content. ([GitHub][1])

---

## 0) Design goals (tight scope)

* **Determinism & auditability:** every sample is reproducible from a **challenge_id**; transcripts and verdicts are externally checkable. Inspired by Procgen’s seeded generation and Gymnasium’s Step API. ([GitHub][2])
* **Minimalism:** small surface area, few files, few deps, short functions.
* **Exploit resistance:** blind, deterministic generation; strict schemas; signed blocks; weight copying resistant by aligning with Bittensor’s commit-reveal & VTrust. ([Taostats Documentation][3])

---

## 1) Repository layout (fresh tree)

```
affine/
  core/
    types.py           # dataclasses for Challenge, Sample, Verdict, BlockHeader
    rng.py             # seed->np.random.Generator (PCG64)
    wilson.py          # Wilson CI helpers
    hashing.py         # blake3/sha256, block hashing
  envs/                # Gymnasium envs
    tictactoe.py       # multi-turn, deterministic positions from challenge_id
    mult8.py           # single-turn, 8-digit × 8-digit
    base.py            # BaseEnv mixin: emits challenge_id, metadata, verify()
  duel/
    local.py           # contend(champ, cont) on one env, sequential stopping
    aggregate.py       # multi-env early stopping, geometric ratio-to-beat
  validators/
    sampler.py         # independent sampling loop, sends queries to chutes
    blocks.py          # build+sign block of samples, prev_hash chaining
    merge.py           # pull peers’ blocks, verify, build global scoreboard
    weights.py         # compute winner-takes-all weights, push to chain
  miners/
    chutes_client.py   # minimal client to call a model chute (request_id capture)
  cli.py               # af validate / af duel / af env run / af set-weights
  pyproject.toml
```

**Notes**

* Use **Gymnasium** (`gymnasium`), not legacy `gym`. Supports `seed` and the modern `terminated|truncated` API. ([Gymnasium][4])
* Determinism & speed inspired by Procgen: per-challenge **seed → episode** mapping. ([GitHub][2])
* Keep Bittensor integration minimal: only **read miners**, **set weights**, respect **Yuma/commit-reveal** and **VTrust** semantics. ([docs.learnbittensor.org][5])

---

## 2) Environments (Gymnasium, deterministic, externally verifiable)

### Common contract (`envs/base.py`)

* `reset(seed: int | None) -> (obs, info)` where `info['challenge_id'] = hexseed`
* `metadata = {'env_id': 'tictactoe-v0', 'spec_version': 1}`
* `verify(answer, info) -> Verdict` (pure function) returns `{ok: bool, reason: str}`
* Procedural generation: `seed = H(env_id || challenge_id || env_spec_version)` → `np.random.Generator(PCG64(seed))`. ([GitHub][2])
* All randomness lives in the environment; **no network calls** in envs.

### `tictactoe-v0` (multi-turn)

* Deterministic puzzle **positions** (some “win in k / don’t-lose” setups) generated from the seed; the agent plays one side to episode end against a **perfect minimax** opponent (tiny, 3×3). Reward = +1 win, 0 draw, −1 loss; success is **win or draw** depending on spec. (Minimax is <100 lines.)
* Observation: flat 9-cell board; Action: `Discrete(9)`; `terminated` when win/loss/draw.
* `verify()` reconstructs the same start state and checks the action sequence; no ambiguity in scoring.

### `mult8-v0` (single-turn)

* Prompt: `“Compute A × B; return only the integer.”` Numbers A,B come from the seed (8 digits, allow leading non-zero). Ground truth is exact integer.
* `verify()` extracts the **last integer token sequence** from the reply (or require JSON schema if model supports function calling), compares to truth.

**Why Gymnasium?** Stable API, clear seeding (`env.reset(seed=...)`) and modern Step semantics for turn handling. ([Gymnasium][4])

---

## 3) Duel evaluation (contender vs current best only)

We stop doing round-robin and Pareto. Affine becomes a **king-of-the-hill duel machine**:

### 3.1 Single-env duel with sequential Wilson stopping

Maintain `(wins, total)` for **contender** across challenges of one env.

* Use **Wilson score interval** (or Jeffreys as a drop-in) for binomial success rate `p`. We stop when either:

  * `lower_bound(p) > ratio_to_beat` ⇒ **contender beats champ** on this env; or
  * `upper_bound(p) < ratio_to_beat` ⇒ **champ holds** on this env; or
  * a **max budget** is hit (failsafe).
    Default: 95% CI; Wilson chosen for small-sample robustness. ([Wikipedia][6])

Pseudo (tight):

```python
def duel_env(stream_results, ratio_to_beat=0.5, z=1.96):
    wins = total = 0
    for ok in stream_results:     # ok=True if contender wins this challenge
        total += 1; wins += int(ok)
        lo, hi = wilson_ci(wins, total, z)
        if lo > ratio_to_beat:  return ("contender", wins, total, lo, hi)
        if hi < ratio_to_beat:  return ("champion",  wins, total, lo, hi)
    return ("inconclusive", wins, total, lo, hi)
```

### 3.2 Multi-env aggregation with early stopping

There are `N` environments. We **process envs in parallel** (async) and **early-stop** globally:

* Per env, run the above sequential test until it yields a side.
* Track `env_wins_cont`, `env_wins_champ`, `env_left`. Stop as soon as one side **cannot** reach `(N+K)/(2N)` env wins (i.e., a strict majority threshold with margin **K**). This saves cost: if the contender loses early `0/N`, we stop.
* When a contender wins, **record the per-env empirical win rates** used at stopping time (the Wilson-consistent MLE `wins/total`).

### 3.3 Ratio-to-beat schedule (anti-trivial dethronements)

* Initial `ratio_to_beat = 0.5 + ε`.
* When a contender dethrones, set the new target to the **geometric mean** of the per-env win rates from the victory (capped ≤0.95), so future challengers must **beat or match** that bar.
* Apply **exponential decay** toward 0.5 over wall-time or epochs:
  `ratio_to_beat ← 0.5 + (ratio_to_beat - 0.5) * exp(-λ · Δtime)`
  (configurable half-life) to avoid lock-in.

---

## 4) Validator workflow (independent sampling, shared evidence)

Validators act independently but **commit to evidence** and **settle weights together**:

### 4.1 Sampling & evidence (per validator)

For each miner pair (champ, cont):

* **Choose seeds** independently (e.g., `challenge_id = blake3(validator_hotkey || env_id || counter)`).
* Run env(s), query **each miner via Chutes** (capture **request_id** or equivalent trace token; keep full prompt/response transcript). ([GitHub][7])
* Produce **Sample**:

  ```
  {
    env_id, env_spec_version,
    challenge_id,
    miner_id, role: {champ|cont},
    request_id,    # if available from Chutes
    prompt, response, info,          # complete transcript
    ok: bool, reason,                # env.verify()
    timing: {latency_ms}, bytes
  }
  ```
* Write **blocks** of samples (e.g., 100 samples per block):

  * **BlockHeader** = `prev_hash, block_index, ts, validator_hotkey, env_spec_versions, sample_count, merkle_root, signature`.
  * **Block** = header + array of `sample_hashes` + optional embedded samples.
  * Hash with BLAKE3 or SHA-256; **sign header** with hotkey.
  * Push blocks to a **shared bucket** (HTTP/S3/IPFS—keep the interface pluggable).

The block chain forms an append-only, validator-local log with **tamper-evidence** and **ordering**. (Fits Bittensor’s commitment culture and mitigates “weight copying.”) ([Taostats Documentation][8])

### 4.2 Cross-pull & VTrust

Validators **pull each other’s blocks**, verify hashes/signatures, and re-compute `env.verify()` on the attached transcripts:

* Assign **VTrust(v)** from **overlap accuracy**: the Wilson **lower bound** of validator *v*’s “correct-vs-global” ratio on intersecting samples. Penalize invalid blocks, missing transcripts, or unverifiable claims. (Aligns with chain notion that VTRUST reflects how well a validator matches consensus.) ([docs.learnbittensor.org][5])

### 4.3 Weight setting (winner-takes-all)

* If the **global aggregation** (using all validators’ verified samples, weighted by their VTrust) says the **contender wins**, **set champion = contender**.
* **Weights**: winner gets 1.0, others 0.0 (or tiny epsilon for availability). Submit via the **Yuma** interface, letting **commit-reveal** hide near-term weights to deter copying. ([docs.learnbittensor.org][5])

---

## 5) Minimal APIs (crisp, testable)

### 5.1 Core types (dataclasses)

```python
@dataclass(frozen=True)
class Challenge: env_id:str; challenge_id:str; meta:dict

@dataclass(frozen=True)
class Verdict: ok:bool; reason:str=""

@dataclass(frozen=True)
class Sample:
    env_id:str; challenge_id:str; miner_id:str; role:str
    prompt:str; response:str; info:dict
    ok:bool; reason:str; request_id:str|None=None

@dataclass(frozen=True)
class BlockHeader:
    prev_hash:str; ts:int; validator:str; idx:int
    sample_count:int; merkle_root:str; signature:str
```

### 5.2 Envs (Gymnasium)

```python
class AffineEnv(Env):
    def reset(self, seed:int|None=None, options=None): ...
    def step(self, action): ...
    def verify(self, response:str, info:dict) -> Verdict: ...
```

### 5.3 Duel API

```python
def duel_env(miner_cont, miner_champ, env, ratio_to_beat:float)-> dict
def duel_many_envs(..., env_ids:list[str], K:int, ratio_to_beat:float)-> dict
```

### 5.4 Validator APIs

```python
def sample(miner_id:str, env_id:str, challenge_id:str) -> Sample
def build_block(samples:list[Sample], prev_hash:str) -> Block
def merge_and_score(blocks:list[Block]) -> dict  # global scoreboard, VTrust
def set_winner_weights(winner_id:str) -> None
```

---

## 6) Determinism & verification details

* **Challenge IDs:** 128-bit hex. Seed derivation per env:
  `seed = uint64( blake3(env_id || challenge_id || spec_version)[:8] )`.
  Use **NumPy PCG64** for RNG. (Procgen-style practice.) ([GitHub][2])
* **Reproducibility contract:** any third party with `(env_id, spec_version, challenge_id, transcript)` can recompute the start state, re-play actions, and re-score.
* **Transcript strictness:** prefer **structured JSON** answers where possible; otherwise, robust integer extraction for `mult8`.
* **No hidden state:** envs derive everything from the seed; validators only add **who answered what**.

---

## 7) Anti-cheat & failure modes (practical, minimal)

* **No answer leakage:** prompts avoid embedding ground truth; envs compute truth on the validator side.
* **Timing signals:** record `latency_ms` and token counts to detect replay/lookup.
* **Duplicate detection:** same `(env_id, challenge_id, miner_id)` within a window is de-weighted.
* **Block tamper:** invalid signatures / broken `prev_hash` chains are ignored and reduce VTrust.
* **Weight copying:** rely on **commit-reveal 3.0** and **VTrust**; penalize validators whose weights diverge from evidence, reward those aligned. ([Taostats Documentation][3])

---

## 8) Migration & rollout

1. **Freeze** current repo; branch `reboot/alpha`. Note current Affine uses Pareto dominance with round-robin—this plan **removes** that. ([GitHub][1])
2. Implement **envs** first (`tictactoe-v0`, `mult8-v0`) + `verify()`.
3. Implement **single-env duel** (Wilson) then **multi-env aggregator** with early stopping. (Wilson math per Wikipedia/NIST—tiny helper.) ([Wikipedia][6])
4. Implement **validator block** format + **merge & VTrust**.
5. Wire **Chutes** client (capture `request_id`/tracing if available) and **Bittensor set_weights** with commit-reveal aware cadence. ([GitHub][7])
6. Delete legacy codepaths (AgentGym, Pareto, round-robin). Keep “validator buckets & commitments” concept by migrating to hash-chained blocks.
7. Ship a **tiny CLI**: `af env run`, `af duel`, `af validate`, `af set-weights`.

---

## 9) Testing (deterministic, short)

* **Env determinism:** 10k seeds → same states & outcomes across runs.
* **Verifier invariants:** replay transcripts → identical verdicts.
* **Wilson harness:** property tests that (a) accept when `p ≥ target` and (b) reject when `p < target`, measuring average sample counts.
* **End-to-end:** local fake miners: one perfect, one noisy → contender dethrones and ratio_to_beat updates.

---

## 10) What stays aligned with the ecosystem

* **Gymnasium** interface and seeding semantics. ([Gymnasium][4])
* **Procedural generation** à la Procgen (deterministic per seed). ([GitHub][2])
* **Bittensor validator model** (Yuma, VTrust, commit-reveal). ([docs.learnbittensor.org][5])
* **Chutes** remains the inference substrate for miners (SN64). ([Taostats][9])

---

## 11) Trim code further (pragmatic tips)

* Prefer **dataclasses + stdlib**; avoid heavy deps.
* Keep envs pure-python; minimax for Tic-Tac-Toe in one file.
* Single **`wilson_ci(wins, total, z)`** function (~10 lines) from the closed form. ([Wikipedia][6])
* CLI via `argparse` with subcommands; no framework.
* Hash via `hashlib.blake2s` or `blake3` if already present.

---

## 12) Open knobs (defaults you can start with)

* `ε = 0.01`, `z = 1.96`, `K = 1` (needs  > 50% of envs + margin), per-env max 200 challenges before “inconclusive”.
* Ratio-to-beat decay half-life: 7 days (or N tempos).
* Block size: 100 samples; `prev_hash` = `sha256(header||sample_hashes)`; sign header with validator hotkey.

---

### References (selection)

* **Affine today**: current README (Pareto/winner-takes-all narrative, Chutes deploy). ([GitHub][1])
* **Gymnasium API & seeding**. ([Gymnasium][4])
* **Procgen** (deterministic, seeded, Gym-style envs). ([GitHub][2])
* **Chutes** (SN64, deploy/query, hotkey auth). ([Taostats][9])
* **Bittensor validation, VTrust, Yuma, Commit-Reveal**. ([docs.learnbittensor.org][5])
* **Wilson interval** background & formulae. ([Wikipedia][6])

---

If you want, I can draft the exact `wilson_ci()` helper, the `mult8-v0.verify()` (robust integer parse), and the tictactoe **minimax** (≤80 LoC) in this style so you can drop them straight into `core/` and `envs/`.

[1]: https://github.com/AffineFoundation/affine "GitHub - AffineFoundation/affine: Anima Machina"
[2]: https://github.com/openai/procgen?utm_source=chatgpt.com "Procgen Benchmark: Procedurally-Generated Game-Like ..."
[3]: https://docs.taostats.io/docs/commit-reveal-30 "Commit Reveal 3.0"
[4]: https://gymnasium.farama.org/index.html?utm_source=chatgpt.com "Gymnasium Documentation"
[5]: https://docs.learnbittensor.org/validators "Validating in Bittensor | Bittensor"
[6]: https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval?utm_source=chatgpt.com "Binomial proportion confidence interval"
[7]: https://github.com/rayonlabs/chutes "GitHub - chutesai/chutes"
[8]: https://docs.taostats.io/docs/validation "Validator (Architecture)"
[9]: https://taostats.io/subnets/64/chart?utm_source=chatgpt.com "0.0755 · SN64 · Chutes"
