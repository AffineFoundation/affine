# Affine Reboot — Implementation Spec (no-code)

This is a clean-room spec the team can implement directly. It turns your plan into concrete modules, data contracts, algorithms, CLI, and acceptance tests. Keep it small, deterministic, and verifiable. No legacy paths, no round-robin, no Pareto—only king-of-the-hill duels with evidence.

---

## 1) Architecture & Responsibilities

```
affine/
  core/              # pure, dependency-light utilities
    types.py         # dataclasses & enums, canonical JSON helpers
    rng.py           # challenge_id -> numpy.random.Generator(PCG64)
    hashing.py       # blake3/sha256, merkle, canonical json hash
    wilson.py        # Wilson CI (one-sided), sequential stopping helpers
    verify.py        # schema-robust extraction (e.g., last-integer)
  envs/              # Gymnasium environments
    base.py          # AffineEnv mixin: seed plumbing, verify() contract
    tictactoe.py     # deterministic, multi-turn vs. minimax opponent
    mult8.py         # deterministic, single-turn 8×8-digit multiplication
    registry.py      # {env_id: factory}; versioned, immutable specs
  duel/
    single.py        # per-env duel loop with Wilson stopping
    aggregate.py     # multi-env early stop + ratio-to-beat logic/state
  validators/
    sampler.py       # commit→reveal plan, sampling loop, query miners
    blocks.py        # block & header build/sign/verify; prev_hash chain
    merge.py         # pull/verify peers’ blocks; global scoreboard
    vtrust.py        # compute validator trust from overlap correctness
    weights.py       # compute winner & push winner-takes-all weights
  miners/
    chutes_client.py # HTTP client; prompt/response; request_id capture
  cli.py             # af env, af duel, af validate, af set-weights
  pyproject.toml
```

**Dependencies:** `gymnasium`, `numpy`, `blake3` (or stdlib sha256 if missing), `ed25519` for signatures, `requests` (or httpx), `argparse`. Keep dev-deps minimal.

---

## 2) Global Contracts

### 2.1 Identifiers & Versioning

* `env_id`: e.g., `tictactoe-v0`, `mult8-v0`. Increment suffix on *any* spec-affecting change.
* `spec_hash`: blake3 of the environment’s canonical spec bundle (constants + code fingerprint). Included in every sample for third-party reproducibility.
* `challenge_id`: 128-bit hex (lowercase). Generation:
  `seed_bytes = blake3(env_id || spec_hash || validator_hotkey || epoch_anchor || counter)[:8]`
  `seed = uint64_le(seed_bytes)` → PCG64.
* `request_id`: opaque string returned by Chutes (if available) for tracing.

### 2.2 Canonical JSON

* Deterministic serialization: UTF-8, sorted keys, no whitespace beyond single spaces, floats as strings if present.
* `canonical_hash(obj) = blake3(canonical_json(obj))`.

### 2.3 Signatures

* Sign **BlockHeader** with validator hotkey (ed25519).
* Verify on merge; invalid signatures or broken chains → block rejected and penalized in VTrust.

---

## 3) Environments (Gymnasium, deterministic, verifiable)

### 3.1 Base contract (`envs/base.py`)

* `reset(seed: int | None) -> (obs, info)`; **must** set:

  * `info["challenge_id"] = <hex-128>`
  * `info["env_id"]`, `info["spec_hash"]`, `info["difficulty"]` (int; start at 0)
* `metadata = {"env_id": <env_id>, "spec_version": 1}`
* `verify(response: str | dict, info: dict) -> Verdict` (pure; no I/O)
* All randomness comes from `core.rng.make(seed)` (PCG64). **No network calls** inside envs.

### 3.2 `tictactoe-v0`

* **State:** 3×3 board; starting player and (optional) prefilled legal positions are derived from RNG(seed). For v0 you may start from empty board for simplicity.
* **Observation:** flat 9-cell array (0 empty, 1 us, −1 them) or a compact textual board; must be deterministic for a given seed.
* **Action space:** `Discrete(9)` (index 0..8). Illegal moves auto-lose or are rejected (choose and document).
* **Opponent:** perfect minimax (≤80 LoC), no randomness.
* **Reward:** +1 win, 0 draw, −1 loss. Duel **win** = “contender’s result better than champion’s result on the same challenge when each plays the same side.”
* **verify():** deterministically rebuild start state and check the action sequence for legality and outcome; return `Verdict(ok, reason)`.

### 3.3 `mult8-v0`

* **Observation/Prompt:** `"Compute A × B; return only the integer."` where A, B are 8-digit integers from RNG(seed). Include them in `info`.
* **Action space:** free-form text; model returns an answer string.
* **Scoring:** correct iff the **last integer token sequence** equals `A*B` (base 10, no separators). If both miners correct or both wrong → tie (excluded from binomial `n`).
* **verify():** robust integer extraction (`core.verify.last_integer(text)`); exact big-int compare.

**Acceptance (envs):**

* 10k random `challenge_id`s reproduce identical prompts, start states, and truths across processes and machines.
* `verify()` is side-effect free and idempotent.

---

## 4) Duel Engine

### 4.1 Per-env duel (`duel/single.py`)

Maintain `(wins, losses, trials)` for **contender vs champion** on one env. A trial increments when exactly one miner is correct/better on the same `challenge_id`.

* **Stopping test** (one-sided Wilson at 95% by default):

  * Let `p̂ = wins / (wins + losses)`.
  * Compute Wilson lower/upper bounds (`z = 1.96`).
  * Stop with **contender wins** if `lower > ratio_to_beat_env`.
  * Stop with **champion holds** if `upper < ratio_to_beat_env`.
  * Else continue until `max_trials` (failsafe → “inconclusive”).
* **Tie handling:** ties do not change `wins` or `losses` (they do not count toward `trials`).
* **Concurrency:** run multiple `challenge_id`s concurrently (async), but **commit** results in arrival order to preserve sequential test correctness.

**Config (defaults):** `z=1.96`, `ratio_to_beat_env=0.5+ε`, `ε=0.01`, `max_trials=200`.

**Output:**
`{"env_id": str, "result": "contender|champion|inconclusive", "wins": int, "losses": int, "trials": int, "ci": [lo, hi]}`

### 4.2 Multi-env aggregation (`duel/aggregate.py`)

Given env set `E`:

* Launch per-env duels in parallel.
* Maintain `W = envs won by contender`, `L = envs lost`, `U = undecided`.
* **Global ratio-to-beat** `Rglob ∈ (0.5, 1]`. Early stop rules:

  * If `W ≥ ceil(Rglob * |E|)` → contender dethrones.
  * If `W + U < ceil(Rglob * |E|)` → impossible to reach target → champion holds.
  * Else wait for more env results (or cap).

**Ratio-to-beat schedule:**

* Start `Rglob = 0.5 + ε`.
* On dethroning:

  * For each **won** env: `r_e = (wins_e + 1) / (losses_e + 1)`.
  * `g = geometric_mean({r_e})`; map to probability `Rnew = g / (1 + g)`, cap at `0.95`.
  * Set `Rglob_peak = max(Rglob, Rnew)`.
* **Decay:** each epoch (or wall-time), `Rglob = 0.5 + (Rglob - 0.5) * exp(-Δ/τ)` (half-life default 7 days).

**Acceptance (duel):**

* Synthetic Bernoulli streams with true `p` above/below target are accepted/rejected with ≤5% Type-I error in Monte-Carlo tests.
* Average sample count scales as expected (small near extremes, larger near target).

---

## 5) Validators: Sampling, Evidence, Consensus

### 5.1 Commit → Reveal Sampling (`validators/sampler.py`)

* At epoch `T`, publish `commit = blake3(secret_seed_v_T)`.
* After reveal delay, publish `secret_seed_v_T`.
* For each env `e` and local counter `i=0…`, derive:
  `seed_v,T,i,e = blake3(secret_seed_v_T || epoch_anchor_T || env_id || i)`.
* Only **on-schedule** `challenge_id`s are valid for scoring.

**Sampling loop:**

* For `(champ_uid, cont_uid)` pair:

  * Generate `challenge_id`.
  * Build prompt/obs; query **both** miners via Chutes (capture `request_id`).
  * Score with env `verify()`.
  * Emit **Sample** (see schema).

### 5.2 Sample & Block Schemas (`validators/blocks.py`)

**Sample (canonical JSON):**

```
{
  "env_id": "tictactoe-v0",
  "spec_hash": "<hex>",
  "challenge_id": "<hex128>",
  "validator": "<hotkey>",
  "epoch": <int>,
  "best_uid": <int>,
  "cont_uid": <int>,
  "request": {
    "best": {"request_id": "ch_...", "prompt": "..."},
    "cont": {"request_id": "ch_...", "prompt": "..."}
  },
  "response": {
    "best": {"text": "...", "latency_ms": <int>, "bytes": <int>},
    "cont": {"text": "...", "latency_ms": <int>, "bytes": <int>}
  },
  "verdict": {"winner": "best|cont|tie", "reason": "string"},
  "timing": {"started_at": <unix_ms>, "ended_at": <unix_ms>},
  "hash": "b3:<hex32>"
}
```

**BlockHeader:**

```
{
  "prev_hash": "b3:<hex32>" | null,
  "block_index": <int>,
  "ts": <unix_ms>,
  "validator": "<hotkey>",
  "epoch": <int>,
  "env_spec_versions": {"tictactoe-v0":"<spec_hash>", "mult8-v0":"<spec_hash>"},
  "sample_count": <int>,
  "merkle_root": "b3:<hex32>",
  "signature": "ed25519:<base64>"
}
```

**Block:** `{"header": <BlockHeader>, "samples": [<Sample or sample_hash>...]}`
Allow “headers-only” blocks (list of hashes) with samples retrievable by URL to keep blocks small.

**Block rules:**

* `block_hash = blake3(canonical_json(header) || concat(sample_hashes))`.
* `signature = sign(block_hash)`.
* Upload to shared store (HTTP/S3/IPFS). Path includes `block_hash`.

### 5.3 Merge, Verify, Scoreboard (`validators/merge.py`)

* Pull latest heads for peers. Walk prev_hash chain until last known or retention horizon.
* **Verify:**

  * header signature, hash chain, merkle inclusion for inlined samples.
  * each sample `verdict` by recomputing env `verify()` (using included prompts/responses/info).
  * on-schedule: recompute `challenge_id`s from revealed seed; reject off-plan.
  * dedupe by `(env_id, spec_hash, challenge_id, validator)`.

**VTrust (`validators/vtrust.py`):**

* For validator `v`, compute correctness on *intersections*:

  * `correct_v = #samples where local recompute matches posted verdict and is on-schedule`
  * `incorrect_v = #invalid, unverifiable, off-schedule, or mismatched`
  * `vtrust_v = (correct_v + α) / (correct_v + incorrect_v + α + β)` with `α=β=1`.
* Use `vtrust_v` as a weight on **validator contributions**.

**Global duel decision:**

* Reconstruct per-env duel tallies from the union of valid samples, **weighted by vtrust of sample’s validator** (weight the *votes*, not the miner).
* Apply the same stopping/aggregation rules (§4) on the merged stream ordering by `challenge_id` ascending as a canonical merge order (break ties by `(validator_hotkey, block_index)`).

### 5.4 Set Weights (`validators/weights.py`)

* If global decision → **contender wins**, set `champ_uid = cont_uid`.
* Compute network weights: `winner_uid: 1.0`, all others: `0.0` (or `ε_availability` if subnet requires).
* Respect commit-reveal cadence for weight submissions if applicable.
* Expose dry-run flag for testnets.

**Acceptance (validators):**

* Tampering with any in-chain field changes `block_hash` and is detected.
* Off-schedule or unverifiable samples reduce `vtrust` and cannot swing outcomes.
* Independent validators converge to the same winner on the same evidence.

---

## 6) CLI (single file, `argparse`)

* `af env run --env tictactoe-v0 --seed <hex|int> [--render]`
  Runs one episode and prints `challenge_id`, prompt/obs, and ground truth (for dev only).
* `af duel --cont <uid> --champ <uid> --envs tictactoe-v0,mult8-v0 [--z 1.96 --eps 0.01 --max 200]`
  Local duel (no blocks).
* `af validate --cont <uid> --champ <uid> --epochs 1 --bucket <s3|http url> [--commit|--reveal]`
  Produces blocks, uploads, prints head hash.
* `af set-weights --buckets <urls.json> [--dry-run]`
  Merges, decides winner, and sets weights (or prints the transaction payload in dry-run).

---

## 7) Config & Defaults

* **Environment variables:**
  `AFFINE_BUCKET_URL`, `AFFINE_VALIDATOR_HOTKEY`, `AFFINE_NETWORK`, `AFFINE_RATIO_HALF_LIFE_DAYS=7`, `AFFINE_Z=1.96`, `AFFINE_EPS=0.01`, `AFFINE_MAX_TRIALS=200`.
* **Timeouts:** per request 30s, per mult8 verify 1s, per ttt move 2s.
* **Block size:** 100 samples; rotate early at 10 MB.
* **Retention:** keep last 10k blocks per validator; GC older (but never rewrite).

---

## 8) Testing & Acceptance

### 8.1 Unit / Property tests

* **Determinism:** 10k random `challenge_id`s → identical env starts and truths across runs.
* **Verify invariants:** any transcript replay yields identical verdicts.
* **Wilson math:** property tests vs. brute-force Monte-Carlo; symmetry checks.
* **Canonical JSON:** different key orders hash to same digest.

### 8.2 Integration

* **Fake miners:** one perfect, one noisy (`p = 0.55`) → dethrone at `Rglob≈0.51`.
* **Edge:** contender `p = 0.49` → champion holds with high probability; sample counts near max.
* **Blocks:** corrupt signature/prev_hash → merge rejects; vtrust reduced.

### 8.3 End-to-End

* Two validators run independently with same commit→reveal schedule; merged scoreboard yields identical winner and identical weight payload bytes.

**Exit criteria for “alpha”:**

* Both envs pass determinism & verify tests.
* Duel engine selects correct winner across 1k simulated runs with ≤5% false decision rate at boundary `p≈R`.
* Blocks/vtrust prevent a single malicious validator from flipping outcome when ≥1 honest validator participates.

---

## 9) Security & Anti-Cheat

* **No truth leakage:** prompts never include the computed ground truth.
* **On-schedule only:** samples must match revealed schedule; out-of-schedule ignored + penalized.
* **Duplicate suppression:** unique key `(env_id, spec_hash, challenge_id, miner_id)`; dupes de-weighted.
* **Timing:** record latencies & token bytes; expose anomaly metrics (e.g., zero-latency bursts).
* **Key handling:** hotkeys read from secure keystore; never logged; signatures verified on merge.

---

## 10) Migration Plan

1. Branch `reboot/alpha`; freeze main.
2. Hard delete legacy: AgentGym, Pareto, round-robin, non-deterministic envs.
3. Implement `core/`, `envs/`, `duel/` minimal set.
4. Implement validator blocks + merge + vtrust + weights.
5. Wire Chutes client + Bittensor weights.
6. Ship CLI; run a 48-hour dry-run (no on-chain weights) across two validators.
7. Enable weights; announce changeover; deprecate old CLI.

---

## 11) Developer Checklists

### 11.1 Env Author Checklist

* [ ] All randomness from `rng.G(seed)` only.
* [ ] `info` includes `challenge_id`, `env_id`, `spec_hash`, `difficulty`.
* [ ] `verify()` is pure and idempotent.
* [ ] Spec bump and new `spec_hash` on any behavior change.
* [ ] Unit tests: determinism (10k seeds), verify invariants.

### 11.2 Duel/CI Checklist

* [ ] `wilson.lower/upper` matches reference numeric tests.
* [ ] Ties excluded from trials.
* [ ] Sequential stop honored with concurrent execution.
* [ ] Global early-stop triggers at the right bounds.

### 11.3 Validator Checklist

* [ ] Commit→reveal implemented; off-plan samples rejected.
* [ ] Block signatures verified; merkle root recomputed.
* [ ] vtrust diminishes on any unverifiable/mismatched sample.
* [ ] Merge produces same winner on repeated runs.

---

## 12) Pseudocode (spec, not code)

### 12.1 Wilson CI (one-sided)

```
def wilson_ci(wins, trials, z):
    if trials == 0: return (0.0, 1.0)
    p = wins / trials
    z2 = z*z
    denom = 1 + z2/trials
    center = p + z2/(2*trials)
    margin = z * sqrt( (p*(1-p)/trials) + (z2/(4*trials*trials)) )
    lo = (center - margin) / denom
    hi = (center + margin) / denom
    return (max(0, lo), min(1, hi))
```

### 12.2 Per-env duel loop

```
wins = losses = 0
for ok in stream_results():          # yields: 'cont' | 'best' | 'tie'
    if ok == 'cont': wins += 1
    elif ok == 'best': losses += 1
    trials = wins + losses
    lo, hi = wilson_ci(wins, trials, z)
    if lo > ratio_to_beat: return WIN
    if hi < ratio_to_beat: return LOSS
    if trials >= max_trials: return INCONCLUSIVE
```

### 12.3 Canonical sample hash

```
body = canonical_json(sample_without_hash)
sample.hash = "b3:" + blake3(body)
```

---

## 13) Metrics & Observability

* **Counters:** samples_ok, samples_tie, samples_offplan, blocks_built, blocks_rejected.
* **Gauges:** vtrust_per_validator, Rglob_current, env_trial_counts.
* **Timers:** latency_ms per miner/env, merge_duration_ms.
* **Logs (structured JSON):** include `challenge_id`, `env_id`, `validator`, `winner`, and reasons.

---

## 14) Open Knobs (ship defaults; make configurable)

* `ε = 0.01`, `z = 1.96`, `max_trials = 200`
* Global `Rglob_max = 0.95`, half-life `τ = 7 days`
* Block size = 100 samples
* Timeouts: request 30s; ttt move 2s; mult8 verify 1s

---

## 15) Glossary

* **Champion:** current best miner.
* **Contender:** challenger miner under test.
* **Trial:** a non-tie head-to-head outcome on a single `challenge_id`.
* **Rglob:** environment-level majority threshold (winner must win ≥ `ceil(Rglob*|E|)` envs).
* **VTrust:** validator trust from overlap correctness; used to weight evidence.

---

### Final Notes for Devs

* Keep functions short and pure. Prefer dataclasses + stdlib. Avoid cleverness.
* All “truth” must be recomputable from `(env_id, spec_hash, challenge_id, transcript)`.
* If you add an env, you add one file and one `verify()`—nothing else changes.
* If you change behavior, bump the env version and `spec_hash`; never hot-edit a spec in place.

This spec is intentionally tight. If you implement exactly this, you’ll get a minimal, verifiable, winner-takes-all subnet with clear evidence trails and robust early stopping—while staying friendly to Gymnasium and Bittensor/Chutes.
