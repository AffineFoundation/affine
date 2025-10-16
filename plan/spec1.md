# Affine Reboot — Implementation Spec (no code)

This is a concrete, low-surface-area spec the team can implement directly. It keeps Affine on **Bittensor** (SN64 miners run on **Chutes**) and rebuilds the project around: deterministic **Gymnasium** environments, **duel-only** evaluation with Wilson sequential stopping, and **independent validator sampling** with signed, hash-chained sample blocks and **VTrust-weighted** winner-takes-all weights.

---

## 0) Scope & Goals (binding)

**Must:**

* Determinism & auditability: every sample must be reproducible from a `challenge_id`; any third party can recompute verdicts.
* Minimalism: few files, small functions, standard library where possible.
* Exploit resistance: blind deterministic generation; strict schemas; signed blocks; resistance to “weight copying” via commit-reveal & VTrust.

**Non-goals:**

* No round-robin or Pareto dominance.
* No large dependency trees, no learned judges, no training loops.

**Target platform:** Python ≥3.11, Gymnasium, NumPy (PCG64), blake3 (or hashlib fallback), ed25519 (hotkey signing), requests/httpx, argparse.

---

## 1) Repository Layout

```text
affine/
  core/
    types.py           # dataclasses: Challenge, Sample, Verdict, BlockHeader, DuelResult
    rng.py             # seed -> numpy.random.Generator(PCG64)
    wilson.py          # Wilson CI (one function) + helpers
    hashing.py         # blake3/sha256 helpers; canonical JSON encoding; Merkle root
  envs/
    base.py            # AffineEnv mixin: emits challenge_id, metadata, verify()
    tictactoe.py       # multi-turn; perfect minimax opponent; deterministic from seed
    mult8.py           # single-turn; 8-digit × 8-digit; exact judge
    registry.py        # name->factory; versioning; spec_hash
  duel/
    local.py           # duel on one env (sequential Wilson stopping)
    aggregate.py       # multi-env orchestration & early stopping; ratio-to-beat
  validators/
    sampler.py         # independent sampling loop; calls chutes; captures request_id
    blocks.py          # build/sign blocks; prev_hash chaining; push/pull adapters
    merge.py           # verify peers; recompute verdicts; build global scoreboard
    vtrust.py          # compute validator trust from overlap correctness
    weights.py         # compute & submit winner-takes-all weights to chain
  miners/
    chutes_client.py   # thin client for model chutes (hotkey auth; request_id capture)
  cli.py               # af env run / af duel / af validate / af set-weights
  pyproject.toml
```

**Definition of done (DoD):** each module ships with unit tests, docstrings, and ≤200 LoC per file (guideline, not a hard gate).

---

## 2) Deterministic Environments (Gymnasium)

### 2.1 Common contract — `envs/base.py`

* Inherit Gymnasium `Env`.
* `reset(seed: int | None)` returns `(obs, info)` and **must** set:

  * `info['challenge_id']` (hex, 128-bit)
  * `info['env_id'] = 'tictactoe-v0'` (or `mult8-v0`)
  * `info['spec_version'] = 1` (increment on breaking changes)
  * `info['spec_hash']` = blake3 hash over canonicalized env spec (env code path + parameters)
* RNG:
  `seed64 = uint64(blake3(env_id || challenge_id || spec_version)[:8]); rng = np.random.Generator(PCG64(seed64))`
* Pure **verify** API (no network, no randomness):

  ```python
  def verify(answer: str | list[int] | dict, info: dict) -> Verdict  # pure, deterministic
  ```
* **All** randomness isolated to environment; no I/O in envs.

### 2.2 `tictactoe-v0` (multi-turn)

* Board: 3×3; observation = flat 9 ints in {-1,0,1} or text; action = `Discrete(9)`.
* Start state: deterministic from `rng` (either empty or seeded “tactical” starts).
* Opponent: perfect **minimax** (≤80 LoC).
* Reward: +1 win, 0 draw, −1 loss; but duel scoring collapses to **win/lose/tie** for contender vs champ.
* `verify(transcript, info)` rebuilds start state and replays actions to final board; produces `{ok, reason}` for who won.

### 2.3 `mult8-v0` (single-turn)

* Prompt template: `"Compute A × B; return only the integer."`
  A,B ∈ [10,000,000 .. 99,999,999], drawn from `rng`.
* `verify(reply, info)`: extract final integer sequence (or parse structured JSON if provided) and compare to `A*B`. Returns `{ok, reason}`.

**Acceptance tests:**

* 10k random `challenge_id`s per env yield identical prompts and ground truths across runs/platforms.
* Verifier replays produce identical verdicts for fixed transcripts.

---

## 3) Duel Engine

### 3.1 Single-env sequential test — `duel/local.py`

**Goal:** Decide if **contender** beats **champion** on a given env with high confidence.

**State:** `(wins, total)` counting only decisive samples (ties ignored).

**Stopping rule (Wilson 95% CI):**

* Compute Wilson lower/upper bounds for `p = wins/total`.
* Stop on first `total ≥ 1` where:

  * `lower > ratio_to_beat_env` ⇒ **contender wins this env**, or
  * `upper < ratio_to_beat_env` ⇒ **champion holds**, or
  * `total == max_budget` ⇒ **inconclusive**.

**API (no real code; signature & contract only):**

```python
def duel_env(
    stream_results: Iterable[bool|None],  # True=contender win, False=champ win, None=tie
    ratio_to_beat_env: float = 0.5 + ε,
    z: float = 1.96,
    max_budget: int = 200,
) -> dict  # {env_id, outcome: 'contender'|'champion'|'inconclusive', wins, total, ci:(lo,hi)}
```

**Stream semantics:** caller yields outcomes as each challenge completes; ties yield `None`.

### 3.2 Multi-env aggregation — `duel/aggregate.py`

* Run `duel_env` for each env **in parallel** (async).
* Maintain `env_wins_cont`, `env_wins_champ`, `env_left`.
* **Global early stop:** stop as soon as one side cannot reach
  `ceil(R_global * N_envs)` wins.
* On victory, record per-env empirical `wins/total` at stop time for ratio update.

**API:**

```python
def duel_many_envs(
  contender_uid:int, champion_uid:int, env_ids:list[str], K:int, ratio_to_beat_global:float
) -> dict  # {winner_uid, per_env:{...}, samples_used:int, stopped_early:bool}
```

### 3.3 Ratio-to-beat schedule

* Start `ratio_to_beat_global = 0.5 + ε` (ε=0.01).
* On dethronement:

  * For envs the contender **won**, compute win ratios `r_e = wins_e/(wins_e+losses_e)` (ties excluded), regularized if desired.
  * Set new target to `min(0.95, geometric_mean({r_e}))`.
* Apply exponential decay towards 0.5 over wall-time or epochs:
  `R(t) = 0.5 + (R_peak - 0.5) * exp(-λ · Δt)` (default half-life 7 days).

---

## 4) Validator Workflow

### 4.1 Independent sampling — `validators/sampler.py`

For each pair `(champ, cont)` and each env:

1. **Pick seeds** independently (e.g.,
   `challenge_id = hex(blake3(validator_hotkey || env_id || counter)[:16])`).
2. Run env(s); query miners via **Chutes**; capture **`request_id`** or trace token.
3. Produce **Sample** (strict schema below).
4. Buffer samples into **blocks** of size `B` (default 100); build and sign block; push to bucket.

### 4.2 Sample & Block schemas — `core/types.py`

**Canonical JSON rules:** UTF-8, sorted keys, integers for all counts, ISO-8601 UTC for timestamps (or epoch ms), no NaNs, no floats for indices.

**Sample (immutable):**

```json
{
  "version": 1,
  "env_id": "tictactoe-v0",
  "env_spec_version": 1,
  "challenge_id": "0x7f...ab",
  "validator": "hotkey_v",
  "miner_id": "uid_31",
  "role": "contender",             // or "champion"
  "request_id": "chutes-req-...?", // optional
  "prompt": "...",                 // full prompt string
  "response": "...",               // model raw reply
  "info": {...},                   // env info from reset(); include spec_hash
  "ok": true,                      // from env.verify()
  "reason": "win|loss|draw|timeout|parse_error",
  "timing_ms": 812,                // end-to-end latency
  "bytes": 1234,                   // payload size tracked by client
  "sample_hash": "b3:..."          // blake3 over canonicalized sample minus this field
}
```

**Block:**

```json
{
  "version": 1,
  "header": {
    "prev_hash": "b3:...",         // "" for genesis
    "block_index": 42,
    "timestamp": 173...,
    "validator": "hotkey_v",
    "env_spec_versions": {"tictactoe-v0":1,"mult8-v0":1},
    "sample_count": 100,
    "merkle_root": "b3:...",
    "signature": "ed25519:..."     // sign blake3(header minus signature)
  },
  "samples": ["b3:hash1", "b3:hash2", "..."], // hashes only
  "embedded": []                   // optional: inlined Sample objects for gossip
}
```

**Hashing:** prefer blake3; fallback sha256 with different prefix `sha256:`.
**Merkle:** simple pairwise hash; odd leaf duplicated.

### 4.3 Cross-pull & verification — `validators/merge.py`

* Pull peers’ latest blocks (HTTP/S3/IPFS pluggable).
* Verify:

  * Header signatures, `prev_hash` chain, Merkle proof if samples embedded elsewhere.
  * Re-compute `env.verify()` on embedded samples (or fetch by hash).
* Build **global scoreboard**: for each `(challenge_id, env)`, reconcile `cont vs champ` outcomes.

### 4.4 VTrust — `validators/vtrust.py`

* For each validator `v`:

  * On overlapping samples, compute accuracy vs consensus (recomputed locally).
  * Use Wilson lower bound (or Beta(1,1) posterior mean) as **`VTrust(v)`**.
  * Penalize: invalid signatures, broken chains, unverifiable claims, off-schedule duplicates.

### 4.5 Weight setting — `validators/weights.py`

* Aggregate duel result across all validated samples, **weighted by VTrust** of the contributing validator.
* If **contender wins globally**, set `champion = contender`.
* Submit **winner-takes-all** weights: winner `1.0`, others `0.0` (or tiny ε for availability). Respect **commit-reveal** cadence (Yuma) to blunt weight copying.

---

## 5) Miner Calls — `miners/chutes_client.py`

* Minimal HTTP client:

  * `invoke(uid, prompt, timeout_s) -> (response, request_id, tokens_in/out, latency_ms)`
  * Attach validator hotkey and `challenge_id` as headers if supported.
* Enforce per-env timeouts (defaults: mult8 10s; ttt 2s/move with overall cap).

---

## 6) CLI — `cli.py`

* `af env run --env tictactoe-v0 --challenge-id 0x...`
  Debug single env, print prompt, ground truth metadata.
* `af duel --cont 31 --champ 12 --envs tictactoe-v0,mult8-v0 --k 1`
  Runs full duel locally; prints winner and CI summaries.
* `af validate --cont 31 --champ 12 --push s3://bucket/prefix --block-size 100`
  Validator loop: sample, verify, block, push.
* `af set-weights --from s3://bucket/prefix --dry-run/--commit`
  Pull blocks, merge+score, compute VTrust, set weights (optionally dry-run).

All commands accept `--json` to emit machine-readable outputs.

---

## 7) Algorithms & Math (reference, no code)

### 7.1 Wilson CI (two-sided; used for bounds)

For wins `w`, trials `n`, `p̂ = w/n`, `z = 1.96` (95%):

```
den = 1 + z^2/n
center = p̂ + z^2/(2n)
radius = z * sqrt( p̂(1-p̂)/n + z^2/(4n^2) )
lower = (center - radius)/den
upper = (center + radius)/den
```

### 7.2 Global early stop

Let `E = |envs|`, `R_global ∈ (0.5, 1]`.
Stop if either:

* `env_wins_cont ≥ ceil(R_global * E)` ⇒ contender wins, or
* `env_wins_cont + env_left < ceil(R_global * E)` ⇒ impossible for contender; champ holds.

---

## 8) Config & Defaults

```yaml
confidence_z: 1.96
epsilon: 0.01
ratio_global_half_life_days: 7
per_env_max_budget: 200
block_size: 100
hash_algo: blake3
rng_algo: numpy.PCG64
timeouts:
  mult8_total_s: 10
  ttt_per_move_s: 2
```

All tunables live in `pyproject.toml`’s `[tool.affine]` table or an `.affine.toml`.

---

## 9) Testing & QA

**Unit (fast):**

* `envs`: determinism (same `challenge_id` → same prompt/ground truth); `verify()` invariants.
* `wilson`: property tests accept when `p ≥ target`, reject when `p < target`.
* `hashing`: canonical JSON stable across dict orderings; Merkle root reproducible.
* `blocks`: signature & chain verification failure modes.

**Integration (local):**

* Fake miners: one perfect, one 55% accurate ⇒ contender dethrones at `R≈0.5+ε`.
* Multi-env early stop: verify sample savings when losing early.
* VTrust: inject one faulty validator and confirm down-weighting.

**Soak (overnight):**

* 10k samples per env; zero nondeterminism drift; memory & disk stable; block chain grows linearly.

**Acceptance (DoD per milestone):**

* A1 Envs pass determinism + verify replay.
* A2 Duel single-env converges at expected `n` for synthetic Bernoulli streams.
* A3 Aggregator halts early with correct global decision.
* A4 Blocks and signatures validate end-to-end across two validator processes.
* A5 Weights submit dry-run and live (testnet) without errors.

---

## 10) Security & Anti-Cheat

* **No answer leakage:** prompts never contain ground truth; judge recomputes locally.
* **Strict transcripts:** store raw prompt/response; normalize whitespace for hashing only; include lengths/timings (`latency_ms`, token counts where available).
* **Duplicate suppression:** same `(env_id, challenge_id, miner_id)` within a rolling window counts once.
* **Timeouts:** timeouts count as losses for the timed-out side.
* **Block tamper-evidence:** invalid signatures or broken `prev_hash` ⇒ block ignored and validator’s VTrust penalized.
* **Weight copying:** rely on commit-reveal cadence and VTrust to penalize validators whose weight actions diverge from verified evidence.

---

## 11) Migration Plan

1. **Branch** `reboot/alpha`; freeze main.
2. **Remove** legacy: AgentGym, Pareto/round-robin codepaths, non-deterministic envs, old validators.
3. **Add** `core/`, `envs/`, `duel/`, `validators/`, `miners/`, `cli.py`.
4. **Ship** `tictactoe-v0`, `mult8-v0`, and `verify()` implementations.
5. Wire **Chutes** client (capture `request_id`) and **Bittensor set_weights** with commit-reveal cadence.
6. **Dry-run** network (no on-chain weights) for 48h; review blocks & VTrust.
7. **Enable** live weight setting; monitor first 3 epochs; publish docs.

---

## 12) Precise Interfaces (copy-paste contracts)

### 12.1 dataclasses — `core/types.py` (names & fields only)

* `Challenge { env_id: str, challenge_id: str, meta: dict }`
* `Verdict { ok: bool, reason: str = "" }`
* `Sample { env_id, env_spec_version, challenge_id, miner_id, role, request_id?, prompt, response, info, ok, reason, timing_ms:int, bytes:int, sample_hash }`
* `BlockHeader { prev_hash, timestamp:int, validator:str, block_index:int, sample_count:int, merkle_root:str, signature:str }`
* `DuelResult { outcome:str, wins:int, total:int, ci:(float,float) }`

### 12.2 env base

```python
class AffineEnv(Env):
    metadata = {"env_id": "tictactoe-v0", "spec_version": 1}
    def reset(self, seed:int|None=None, options=None) -> tuple[Obs, dict]: ...
    def step(self, action) -> tuple[Obs, float, bool, bool, dict]: ...
    def verify(self, response:str|dict|list, info:dict) -> Verdict: ...
```

### 12.3 duel APIs

```python
def duel_env(...)->DuelResult
def duel_many_envs(cont_uid:int, champ_uid:int, env_ids:list[str], K:int, ratio_to_beat:float)->dict
```

### 12.4 validator APIs

```python
def sample(miner_id:str, env_id:str, challenge_id:str) -> Sample
def build_block(samples:list[Sample], prev_hash:str) -> dict  # {header, samples, embedded?}
def merge_and_score(blocks:list[dict]) -> dict  # global scoreboard, VTrust per validator
def set_winner_weights(winner_uid:int) -> None
```

---

## 13) Operational Notes

* **Logging:** JSON logs with `ts, level, mod, event, uid, challenge_id, env, latency_ms`.
* **Metrics:** Prometheus counters/gauges: samples_total, ties_total, wins_total, wilson_n, env_stop_reason, vtrust_value, block_height.
* **Storage:** Blocks are append-only; retain forever or prune with snapshot manifests if needed.
* **Performance budgets:** `af validate` under 250 MB RSS; per-sample end-to-end < 1s for mult8 under local latency.

---

## 14) Open Knobs (start values)

* `ε = 0.01`, `z = 1.96`, `K = 1`, per-env `max_budget = 200`.
* Ratio-to-beat decay half-life: 7 days.
* Block size: 100 samples.
* Hash: `blake3`; signature: ed25519 with validator hotkey.

---

### Build Checklist (copy into the PR)

* [ ] `core/` complete: RNG, hashing (canonical JSON + blake3), Wilson, dataclasses.
* [ ] `envs/` complete: `tictactoe-v0` + perfect minimax; `mult8-v0`; both pass determinism tests.
* [ ] `duel/` complete: single-env sequential test + multi-env aggregator + ratio schedule.
* [ ] `validators/` complete: sampler, blocks (sign/verify), merge (verify peers), VTrust, set-weights.
* [ ] `miners/chutes_client.py` calls and captures `request_id`.
* [ ] `cli.py` with four subcommands; `--json` everywhere.
* [ ] CI: unit + integration; publish coverage.
* [ ] Docs: README (how to add an env; how to run a validator); schema docs (Sample/Block).

---

That’s the full implementation spec. If you want, I can also draft **schemas as JSON Schema files** and a **minimal pseudo-code appendix** for `verify(mult8)` and `tictactoe minimax` that you can drop straight into tickets.
