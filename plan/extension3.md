Below is a tightened, implementation‑ready plan to re‑build **Affine** around principled, minimal, and robust primitives—while remaining a Bittensor subnet and keeping the validator/miner roles. I’ve grounded the cuts/keeps in the current repo’s design (Pareto frontier, winners‑take‑all, Chutes integration) and factored in the “agents on Bittensor” inspiration from Ridges. ([GitHub][1])

---

## 0) Design goals (binding)

* **Determinism & verifiability first.** Every sample must be fully reproducible from a `challenge_id` and an environment version.
* **Minimal surface area.** Fewer concepts, fewer files, fewer states; prefer composition over configuration.
* **Asynchronous but simple.** Use `asyncio` for concurrency; avoid complex scheduling logic.
* **Exploit‑resistance.** Unpredictable sampling, commit‑reveal, full transcripts, deterministic judging.
* **One truth.** The same rules must yield the same winner for every validator.

---

## 1) Repository layout (small, sharp)

```
affine/
  core/
    ids.py          # challenge_id spec, hashing, canonical JSON
    prng.py         # stateless PRNG (PCG64/xoshiro) seeded from challenge_id
    types.py        # dataclasses: Sample, Block, Verdict, DuelConfig, etc.
    judge.py        # deterministic judging utils (shared across envs)
    wilson.py       # one-sided Wilson bound + sequential stop
  envs/
    base.py         # Gymnasium adapter: reset(seed)->obs, step(action)
    ttt.py          # TicTacToeEnv (multi-turn, miner vs miner)
    mult8.py        # Mul8Env (single-turn, miner vs answer)
    registry.py     # name->factory (versioned)
  duel/
    arena.py        # contender-vs-best loop per environment (sequential test)
    aggregate.py    # cross-environment early-stop + ratio-to-beat logic
  validator/
    sampleplan.py   # commit→reveal schedule of challenge_ids
    bucket.py       # append-only block store (hash chain) + HTTP/S3 adapter
    vtrust.py       # validator trust from on-chain-verifiable samples
    weights.py      # winner-takes-all weights from shared buckets
  miner/
    client.py       # thin HTTP client to call a miner’s chute (Chutes id)
  net/
    bittensor.py    # set_weights(), uid routing; S120/S64 bindings
    chutes.py       # chute invocation wrapper, capture invocation_id
  cli.py            # af v2 commands
  pyproject.toml
```

> Delete legacy modules outright (AgentGym, Pareto dominance, round‑robin samplers, etc.). Keep only what’s needed for: envs, duels, buckets, vtrust, weights. The current README’s Pareto frontier & Chutes deploy flows are replaced by the duel engine and deterministic envs but remain a subnet with winners‑take‑all semantics. ([GitHub][1])

---

## 2) Environments (Gym; deterministic; externally verifiable)

### 2.1 Common interface (Gymnasium)

* **API:** `env = registry.make(name, version); obs, info = env.reset(seed); obs, reward, terminated, truncated, info = env.step(action)`
* **Determinism:** seed = `H(challenge_id)`; all randomness comes from `core.prng` using that seed.
* **Metadata:** `info` always includes:

  * `challenge_id`, `env` (name@version), `difficulty`, `spec_hash` (hash of env code+params), `ground_truth_commitment` (hash of the judge inputs).
* **Transcript fields** (for each step): `t`, `role` (`env|minerA|minerB`), `content`, `obs_summary`, `action_summary`.

### 2.2 `challenge_id` (portable & strong)

* Format: `cid = base58( blake3( env_name || env_version || epoch_anchor || validator_id || index ) )`
* **Epoch anchor** = hash of a public beacon (e.g., latest finalized block on Bittensor or a recent Bitcoin header); revealed in commit‑reveal (see §5).
* **Fully reproducible:** anyone recomputing `cid` regenerates the same challenge.

### 2.3 Initial envs

**A) Tic‑Tac‑Toe (multi‑turn)**

* **Players:** Miner A vs Miner B; sides rotate per challenge.
* **State:** 3×3 board; optional randomized legal starting board from seed (can be empty for v1).
* **Observation:** Text description of board + “your move”.
* **Action:** `{row,col}` or algebraic `A1..C3`.
* **Terminal reward:** `1` (win), `0` (draw), `-1` (loss) for the acting miner; judge decides outcome from the final board.
* **Win for duel account:** “who won this game”; draws excluded from the Wilson test (do not update n).

**B) 8‑digit multiplication (single‑turn)**

* **Prompt:** “Compute A × B” with two 8‑digit integers from PRNG.
* **Action:** exact product in decimal (canonicalized).
* **Judge:** exact big‑int multiply; **score:** `1` if correct, `0` otherwise.
* **Tie rule:** both correct or both wrong ⇒ discard sample from the binomial count.

> Both envs are fully procedural and deterministic from `challenge_id`, and emit complete metadata for third‑party verification. ([GitHub][1])

---

## 3) Dueling & statistical decision (replace round‑robin)

We only ever compare **Contender** vs **Best**.

### 3.1 Per‑environment, sequential test (one‑sided Wilson)

* Maintain wins `w` = times contender strictly beats best on that environment, and trials `n` (ties not counted).
* Let (\hat p = w/n), (z = \Phi^{-1}(1-\alpha)) (default (\alpha=0.05)).
* Wilson lower bound:
  [
  L = \frac{\hat p + \frac{z^2}{2n} - z \sqrt{\frac{\hat p(1-\hat p)}{n} + \frac{z^2}{4n^2}}}{1+\frac{z^2}{n}}
  ]
* **Stop when** either (L \ge R_{\text{env}}) (contender confidently better) **or** the symmetric test shows best is confidently (\ge R_{\text{env}}) (contender loses). Default (R_{\text{env}} = 0.5 + \epsilon) with (\epsilon=0.01).

### 3.2 Sampling within an environment

* Generate independent challenges by incrementing the per‑env index in `challenge_id` derivation.
* Run games in parallel (async), but **commit results in arrival order** (first‑come sequential analysis) to preserve exact `n`.

---

## 4) Cross‑environment aggregation (global early‑stop)

Let the set of environments be (\mathcal{E}) with (|\mathcal{E}|=E).

* For each env (e), run §3 to get an env‑level verdict: `win`, `loss`, or `undecided` (if bounded by a global cap).
* Maintain `W` = number of envs the contender wins; `L` = losses.
* **Ratio‑to‑beat at env level:** (R_{\text{glob}}\in(0.5, 1]); contender needs at least
  [
  W \ge \lceil R_{\text{glob}} \cdot E \rceil
  ]
* **Early stop across envs:**

  * If (W) already meets the target ⇒ stop (contender dethrones).
  * If (W + (E - W - L) < \lceil R_{\text{glob}}E \rceil) ⇒ impossible to reach target ⇒ stop (contender fails).
  * Else continue to next env (any order; stable round‑robin of envs is fine).

> This aligns with your “(N+K)/(2N)” narrative by setting (R_{\text{glob}}=0.5+K/(2N)) when (E=N). We recommend using (R_{\text{glob}}) directly; it’s simpler and generalizes cleanly.

---

## 5) Ratio‑to‑beat schedule (difficulty ratchet + decay)

Start at (R_{\text{glob}}=0.5+\epsilon). When a contender dethrones the best:

1. For each env (e) **the contender actually won**, compute a **regularized win ratio**:
   [
   r_e = \frac{w_e + 1}{\max(1,, \ell_e) + 1}
   ]
   where (w_e) and (\ell_e) are the contender’s wins/losses vs best within (e) (ties excluded).
2. Geometric mean across winning envs:
   [
   \bar r = \left(\prod_{e \in \text{wins}} r_e\right)^{1/|\text{wins}|}
   ]
3. Map ratio to probability threshold:
   [
   R_{\text{new}} = \frac{\bar r}{1+\bar r}
   ]
   (if (\bar r=1), (R_{\text{new}}=0.5); if (\bar r>1), (R_{\text{new}}>0.5)).
4. **Exponential decay** of the ratchet toward 0.5 to avoid permanent lock‑in:
   [
   R_{\text{glob}}(t) = 0.5 + \big(R_{\text{peak}} - 0.5\big)\cdot e^{-(t-t_{\text{peak}})/\tau}
   ]
   with half‑life (\tau) in validator epochs (default: 14 epochs).

---

## 6) Validators: independent sampling, shared commitments

### 6.1 Sampling plan (commit→reveal)

* At epoch (T), each validator draws a private seed `s_v,T`, publishes **commitment** `C = blake3(s_v,T)`.
* After a fixed delay (e.g., one epoch), reveal `s_v,T`; derive the per‑env `challenge_id`s using:
  [
  \text{seed}*{v,T,i,e} = H(s*{v,T} | \text{epoch_anchor}_T | e | i)
  ]
* **Reject** samples whose `challenge_id` is not on the published schedule (prevents cherry‑picking).

### 6.2 Buckets (append‑only blocks)

* **Block** = canonical JSON with:

  * `prev_hash`, `height`, `created_at`, `validator_id`, `epoch`, `samples[]`, `merkle_root`.
* **Sample** (minimal fields):

  * `challenge_id`, `env`, `difficulty`, `chute_invocation_id`,
    `best_uid`, `contender_uid`, `steps[]` (full transcript),
    `verdict` (`win|loss|tie` from judge), `elapsed_ms`, `hash`.
* **Hashing:** `blake3(canonical_json(sample))`; block hash covers all samples + header.
* **Transport:** plain HTTP(S) or S3‑compatible; path = `/v1/affine/bucket/<block_hash>.json`.
* **Integrity:** validators pull each other’s latest block head and walk backwards (like a simple log chain).

### 6.3 vTrust (validator trust from verifiable samples)

* For each validator (v):

  * `correct` = fraction of samples whose **local recomputation** of the judge matches the posted `verdict` and whose `challenge_id` is on‑schedule and **not duplicated**.
  * `vtrust_v = Beta(α+correct_count, β+incorrect_count)` mean (use α=β=1 for Laplace).
    Use `vtrust_v` as a weight **on the validator**, not on miners.
* **Penalties:** off‑schedule, unverifiable, or missing transcripts ⇒ count as incorrect.

### 6.4 Weight setting (winner‑takes‑all)

* All validators re‑compute the global duel result from the **union of all buckets**, but weight each bucket’s votes by `vtrust_v`.
* If the re‑computed **best_uid** differs from current, switch to the new best; set Bittensor weights to **1.0 for best, 0.0 for others** (respecting the subnet’s API constraints). ([GitHub][1])

---

## 7) Anti‑exploit guardrails (cheap, effective)

* **Deterministic judges**: All correctness checks recompute from `challenge_id`; no subjective labelling.
* **On‑schedule only**: Commit‑reveal ensures validators can’t cherry‑pick or omit loss‑heavy slices.
* **Transcript canonicalization**: strip ANSI, normalize whitespace, limit size (truncate with hash footers).
* **Time budgets**: per‑step and per‑challenge timeouts; timeouts count as losses for the slow side.
* **Tie handling**: remove pressure to overfit by not giving credit for mutual correctness on trivial tasks.
* **Duplicate suppression**: same `challenge_id` from the same validator counts once.
* **Version pinning**: `env` includes `name@semver`; upgrading increments version and changes `spec_hash`.

---

## 8) CLI (one file; friendly)

```
# Validate locally (spawns arena and writes blocks)
af v2 validate --contender <uid> --best <uid> --epochs 1 --push-bucket s3://...

# Publish/reveal sampling plan
af v2 commit-plan --epoch N
af v2 reveal-plan --epoch N

# Recompute winner from buckets + set weights
af v2 set-weights --buckets <urls.json>
```

---

## 9) Defaults (ship with sensible numbers)

* Confidence level (1-\alpha = 0.95), (\epsilon=0.01).
* Per‑env parallelism = CPU count; global cap per env = 5,000 trials.
* Global timeouts: 10s for mult8, 2s per move in ttt; request timeout 30s / sample.
* Ratio‑to‑beat half‑life (\tau=14) epochs.

---

## 10) Testing (short, meaningful)

* **Property tests**: replay determinism (same `challenge_id` ⇒ identical prompts & truth).
* **Sequential test**: synthetic Bernoulli streams with known (p) converge to correct verdict w.h.p.
* **Bucket integrity**: block hash breaks on tampering; merkle proofs verify sample inclusion.
* **End‑to‑end**: contender with (p=0.55) dethrones at (R=0.5+\epsilon); ratchet raises (R), then decays.

---

## 11) Migration & removal plan (concrete)

1. **Branch `v2/cleanroom`.** Freeze main.
2. **Delete** AgentGym, Pareto dominance logic, round‑robin scheduler, and any envs without deterministic judges. (They’re explicitly called out for removal.) ([GitHub][1])
3. **Add** `core/`, `envs/`, `duel/`, `validator/` minimal modules above.
4. **Keep** Bittensor & Chutes wiring (reduced to thin wrappers).
5. **Flip default CLI** to `af v2 …`; leave `af validate` as a shim for one release.
6. **Cut a testnet dry‑run** (no on‑chain weights) for 48 hours; then enable `set-weights`.

---

## 12) Notes on Ridges inspiration

* Keep the **“agents on Bittensor”** posture and a simple **inference gateway** model; avoid complex evaluators. Ridges’ public repos show a similarly thin split of *evaluator*, *inference_gateway*, *validator*—our layout maintains that emphasis but strips to essentials. ([GitHub][2])

---

## 13) Appendix: precise algorithms & schemas

### A) Duel loop (per env; ties dropped)

```python
while True:
    cid = next_challenge_id()
    a, b = play(cid)                 # returns 'contender' or 'best' or 'tie'
    if a == 'contender': w += 1; n += 1
    elif a == 'best':    n += 1
    # else tie: no-op
    L = wilson_lower(w, n, alpha=0.05)
    if n >= n_min and L >= R_env: return WIN
    if n >= n_min and wilson_lower(n-w, n, alpha=0.05) >= R_env: return LOSS
    if n >= n_cap: return UNDECIDED
```

### B) Ratio‑to‑beat update

```
r_e = (wins_e + 1) / (losses_e + 1)
R_new = (gmean(r_e over won envs)) / (1 + gmean(...))
R_glob(t+1) = 0.5 + (R_new - 0.5)          # set peak at dethroning
then decay each epoch: R_glob = 0.5 + (R_glob - 0.5) * exp(-1/τ)
```

### C) Canonical JSON (sample)

```json
{
  "challenge_id": "cid...",
  "env": "ttt@1.0.0",
  "difficulty": 0,
  "chute_invocation_id": "ch_...",
  "best_uid": 12,
  "contender_uid": 31,
  "steps": [{"t":0,"role":"env","content":"..."},
            {"t":1,"role":"minerA","content":"B2"}, ...],
  "verdict": "win",  // from deterministic judge
  "elapsed_ms": 1180,
  "hash": "b3:..."
}
```

---

## What this replaces / preserves

* **Removed:** AgentGym, Pareto dominance/frontier logic, round‑robin scheduling.
* **Preserved:** Subnet identity, Chutes deployments, winners‑take‑all weights; SDK ergonomics become thinner and more reproducible. ([GitHub][1])

---

### Why this meets your bar

* **Principled from the start:** every claim is verifiable from `challenge_id` + env version.
* **Minimalism:** a dozen files; one duel primitive; one aggregation rule; one trust metric.
* **Robust:** randomness anchoring, commit‑reveal, and deterministic judges remove the main exploits.
* **Maintainable:** Gym envs are plug‑and‑play; adding new tasks is just another `envs/*.py`.

If you want, I can turn this into a PR checklist (delete list + add list) and a minimal `ttt.py`/`mult8.py`/`wilson.py` skeleton next.

[1]: https://github.com/AffineFoundation/affine "GitHub - AffineFoundation/affine: Anima Machina"
[2]: https://github.com/ridgesai/ridges "GitHub - ridgesai/ridges: Building Software Agents On Bittensor"
