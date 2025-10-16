# Affine v2 — Implementation Spec (dev-ready, no code)

This spec turns your draft into a concrete, minimal plan the team can build against. It defines scope, architecture, file layout, APIs, algorithms, protocols, CLI, configs, testing, rollout, and guardrails. It purposefully avoids real code while being specific enough to implement directly.

---

## 0) Scope, success criteria, non-goals

### Scope (binding)

* Rebuild Affine as a small, auditable **king-of-the-hill** evaluator:

  1. Deterministic **Gymnasium** environments with verifiable `challenge_id`s.
  2. **Contender vs Champion only** with sequential **Wilson** stopping per env and **early stopping across envs**.
  3. **Independent validators** producing signed, hash-chained **sample blocks**; cross-verified; **VTrust-weighted** global decision; **winner-takes-all** weight setting on Bittensor (SN64 miners run on **Chutes**).
* Keep Bittensor/Chutes integrations **minimal**: read miners, query chutes, set weights.

### Success criteria (acceptance)

* **Determinism:** Given `(env_id, spec_version, challenge_id, transcript)`, any third party reproduces start state and re-scores to the same verdict.
* **Statistical soundness:** In Monte-Carlo tests with known Bernoulli `p`, per-env Wilson stopping rejects/accepts at 95% CI; mean sample counts scale ~O((p−target)⁻²).
* **Inter-validator consistency:** Any validator replays another’s sample block and gets identical `verify()` results for ≥ 99.9% of samples; discrepancies penalized by VTrust.
* **Security:** Tampering with blocks (header or samples) is detected via hash/signature chain; off-schedule samples are ignored; duplicate suppression works.
* **Operational:** CLI can `af env run`, `af duel`, `af validate`, `af set-weights` end-to-end on testnet without manual steps.

### Non-goals

* No Pareto/round-robin/AgentGym.
* No non-deterministic envs, human labelling, or hidden judges.
* No heavy frameworks; keep deps minimal.

---

## 1) Tech baseline

* **Language:** Python 3.11+
* **Core deps:** `gymnasium`, `numpy` (PCG64), `cryptography` (ed25519), `blake3` (optional; fallback to `hashlib.blake2s/sha256`), `httpx` (async).
* **Style:** type-annotated, small modules, <300 LoC per file target.
* **Packaging:** PEP 621 `pyproject.toml`.

---

## 2) Repository layout (fresh tree)

```
affine/
  core/
    types.py           # dataclasses & TypedDicts: Challenge, Sample, Verdict, BlockHeader
    rng.py             # seed -> np.random.Generator(PCG64)
    wilson.py          # Wilson CI + sequential stop helpers
    hashing.py         # blake3/sha256, canonical JSON, merkle root
  envs/
    base.py            # AffineEnv mixin: reset(seed)->(obs,info), verify(response,info)
    tictactoe.py       # tictactoe-v0, deterministic start positions, minimax judge
    mult8.py           # mult8-v0, 8-digit × 8-digit, robust integer extraction in verify()
    registry.py        # env_id -> factory, spec_versioning, spec_hash
  duel/
    local.py           # duel_env(): stream results, Wilson stopping per env
    aggregate.py       # duel_many_envs(): multi-env early stop, ratio-to-beat logic
    schedule.py        # ratio-to-beat decay, persistence
  validators/
    sampler.py         # commit→reveal challenge IDs, query chutes, build Sample
    blocks.py          # BlockHeader, merkle roots, ed25519 sign/verify, prev_hash chaining
    merge.py           # pull peers' blocks, verify, dedupe, recompute verify()
    vtrust.py          # compute validator trust, Wilson-lower-bound or Beta mean
    weights.py         # global scoreboard (VTrust-weighted), winner-takes-all set_weights
  miners/
    chutes_client.py   # minimal async client; captures request_id/trace
  net/
    bittensor.py       # set_weights wrapper (commit-reveal aware)
  cli.py               # `af` subcommands
  pyproject.toml
```

---

## 3) Identifiers, randomness, and canonicalization

### 3.1 `challenge_id` and seeds

* `challenge_id`: 128-bit hex (lowercase), e.g., `8a7b...f3`.
* **Derivation input:** `env_id || spec_version || epoch_anchor || validator_hotkey || counter` → `blake3` digest → take first 16 bytes → hex.
* **Seed for RNG:** `uint64(blake3(env_id || challenge_id || spec_version)[:8])`.
* RNG: `np.random.Generator(np.random.PCG64(seed))`.

### 3.2 Canonical JSON (for hashing/signatures)

* UTF-8, no whitespace changes, sorted keys, no NaN/Infinity, integers as decimals.
* `hash_sample = blake3(canonical_json(sample))`.
* Merkle root: pairwise blake3 over `hash_sample`s.

---

## 4) Core types (no code, exact fields)

### 4.1 Dataclasses

**Challenge**

* `env_id: str` (e.g., `tictactoe-v0`)
* `challenge_id: str` (128-bit hex)
* `meta: dict` (arbitrary, must include `spec_version: int`, `spec_hash: str`)

**Verdict**

* `ok: bool` (did `contender` beat `champion` on this challenge?)
* `reason: str` (short; machine-readable codes preferred)

**Sample**

```
{
  "env_id": "tictactoe-v0",
  "env_spec_version": 1,
  "challenge_id": "8a7b...f3",
  "validator_hotkey": "val...",
  "miner_id": "uid_or_hotkey",
  "role": "champ|cont",
  "prompt": "string",                // exact sent prompt
  "response": "string",              // raw model output
  "info": { ... },                   // env info from reset/step
  "ok": true,                        // result of env.verify()
  "reason": "win|draw|loss|timeout|invalid",
  "request_id": "chutes_trace_id",   // optional
  "timing": { "latency_ms": 1234 },
  "bytes": { "prompt": 321, "response": 1024 },
  "hash": "b3:..."                   // over canonical Sample w/o "hash"
}
```

**BlockHeader**

```
{
  "prev_hash": "b3:...",
  "block_index": 42,
  "ts_unix": 173...,
  "validator_hotkey": "val...",
  "env_spec_versions": { "tictactoe-v0": 1, "mult8-v0": 1 },
  "sample_count": 100,
  "merkle_root": "b3:...",
  "signature": "ed25519:..."         // signature over canonical header w/o signature
}
```

**Block**

```
{
  "header": BlockHeader,
  "sample_hashes": ["b3:...", ...],
  "samples": [Sample, ...]           // optional; if absent, provide per-sample URIs
}
```

---

## 5) Environments (Gymnasium + verify())

### 5.1 Common contract (`envs/base.py`)

* `reset(seed:int|None)->(obs,info)`; `info['challenge_id'] = hexseed`.
* `metadata = {'env_id': 'tictactoe-v0', 'spec_version': 1}`.
* `verify(response:str, info:dict)->Verdict` must be **pure** and recompute truth from seed.
* **No network calls** or global state.
* **Spec hash:** hash of the environment module + constants; emitted as `info['spec_hash']`.

### 5.2 `tictactoe-v0`

* Observation: flat 9 cells (`0,1,-1`) + current player; textual prompt accepted.
* Action: `Discrete(9)`; illegal action ⇒ immediate loss for that move.
* Start position: deterministic from RNG; include empty board in v0 if simpler.
* Opponent: **perfect minimax** (≤100 LoC target) with deterministic tie-breaking.
* Rewards: win=+1, draw=0, loss=−1; **success** = `contender` not losing (win/draw) or stricter “win only” per config (default: win or draw).
* `verify()`:

  * Rebuild start state from `challenge_id`.
  * Parse the action sequence taken by each side from transcripts (or require structured actions via JSON).
  * Replay; if the sequence matches legal transitions and ends in a terminal state, return Verdict with `reason` in `{win,draw,loss}` for contender; else `invalid`.

### 5.3 `mult8-v0`

* Prompt: `"Compute A × B; return only the integer."` with A,B ∈ [10,000,000 … 99,999,999] from RNG.
* `verify()`:

  * **Prefer** JSON tool format if present; else extract the **last integer token sequence** (optional leading `-`, digits only) from response.
  * Compare to exact product; `ok = (parsed == ground_truth)`.
  * Reasons: `win|loss|invalid_parse|timeout`.

---

## 6) Duel evaluation

### 6.1 Per-environment sequential Wilson (ties dropped)

* Maintain `(wins, total)` for **contender** where `total` counts only decisive (non-tie) outcomes.
* Config:

  * `z = 1.96` (95% CI)
  * `ratio_to_beat_env = 0.5 + ε` (ε default 0.01)
  * `n_cap` per env (default 200; configurable)
* Stop when:

  * `lower_bound(wins/total) > ratio_to_beat_env` ⇒ **contender wins env**.
  * `upper_bound(wins/total) < ratio_to_beat_env` ⇒ **champion holds env**.
  * `total >= n_cap` ⇒ **inconclusive**.
* Return record per env: `{winner: 'cont|champ|inconclusive', wins, total, lo, hi}`.

**Pseudocode (reference)**

```
def duel_env(stream_ok: Iterable[bool], target=0.5+eps, z=1.96):
    wins = total = 0
    for ok in stream_ok:           # ok=True if contender beats champ on that challenge
        total += 1; wins += int(ok)
        lo, hi = wilson_ci(wins, total, z)
        if lo > target:  return {"winner":"cont",  "wins":wins, "total":total, "lo":lo, "hi":hi}
        if hi < target:  return {"winner":"champ", "wins":wins, "total":total, "lo":lo, "hi":hi}
        if total >= n_cap: break
    return {"winner":"inconclusive", "wins":wins, "total":total, "lo":lo, "hi":hi}
```

### 6.2 Multi-env aggregation with early stop

* `E = number of envs`.
* Track: `env_wins_cont`, `env_wins_champ`, `env_left`.
* Global threshold: require **strict majority with margin K**: contender must win `≥ ceil((E+K)/(2))` envs. Default `K=1`.
* Early stop conditions:

  * If `env_wins_cont` reaches threshold ⇒ **contender dethrones**.
  * If even winning all `env_left` can’t reach threshold ⇒ **champ holds**.
* Store **per-env empirical win rates** used at stopping to feed ratio-to-beat update.

### 6.3 Ratio-to-beat schedule

* Start `ratio_to_beat_env = 0.5 + ε`.
* On dethroning, set new **env-level bar** to the **geometric mean** of per-env MLEs (`wins/total`) at stopping (capped at `≤0.95`).
* Apply **exponential decay** toward 0.5 over wall-time/epochs:

  * `ratio_to_beat_env ← 0.5 + (ratio_to_beat_env − 0.5) * exp(-λ · Δtime)`
  * Default half-life: 7 days.

---

## 7) Validator workflow

### 7.1 Commit→Reveal sampling

* At epoch `T`, validator draws private seed `s_{v,T}`, publishes **commit** `C = blake3(s_{v,T})`.
* After delay, **reveal** `s_{v,T}`; per env `e` generate `challenge_id[i] = blake3(env_id||spec_version||epoch_anchor_T||hotkey||i)`.

**Rules**

* Only accept samples whose `challenge_id` belongs to a revealed schedule (prevents cherry-picking).
* Duplicate triples `(env_id, challenge_id, miner_id)` are **de-weighted** (first counts; others flagged).

### 7.2 Building & signing blocks

* Bundle samples (e.g., 100 per block).
* Header: `prev_hash, block_index, ts, validator_hotkey, env_spec_versions, sample_count, merkle_root`.
* Sign header with validator hotkey (ed25519).
* Store block as canonical JSON. Push to a shared bucket (HTTP/S3/IPFS; adapter pattern).

### 7.3 Cross-pull & verification

* Periodically pull peers’ latest heads, walk back via `prev_hash`.
* Verify signatures, header consistency, merkle root, sample hashes.
* Recompute `env.verify()` deterministically on embedded transcripts.
* Drop blocks/samples failing any check; record infractions.

### 7.4 VTrust

* For validator `v`, define:

  * `correct_v = #samples that pass schedule, dedupe, verify`
  * `incorrect_v = #samples that fail any of the above`
* Compute **Wilson lower bound** (or Beta mean) for `p = correct/(correct+incorrect)`; call this `VTrust(v)`.
* Use `VTrust(v)` as a **weight on validator evidence** during global merge.

### 7.5 Global merge & weight setting

* Aggregate all verified samples across validators, **weighted by `VTrust(v)`**.
* Re-run duel logic from merged streams; decide **global winner**.
* If **contender wins**, set `weights[winner]=1.0` and others `0.0` (or `ε` for availability if subnet requires).
* Submit via Bittensor **commit-reveal** cadence; schedule submits to reduce weight copying.

---

## 8) Public APIs (signatures; no code)

### 8.1 Envs (Gymnasium)

```
class AffineEnv(Env):
    metadata: dict  # {'env_id': 'tictactoe-v0', 'spec_version': 1}

    def reset(self, seed: int | None = None, options=None) -> tuple[ObsType, dict]: ...
    def step(self, action) -> tuple[ObsType, float, bool, bool, dict]: ...
    def verify(self, response: str, info: dict) -> Verdict: ...
```

### 8.2 Duel

```
def duel_env(miner_cont, miner_champ, env, ratio_to_beat: float, *, n_cap:int=200) -> dict
def duel_many_envs(miner_cont, miner_champ, env_ids: list[str], K:int, ratio_to_beat: float) -> dict
```

### 8.3 Validator

```
def make_challenge_ids(env_id:str, spec_version:int, epoch_anchor:str, hotkey:str, count:int) -> list[str]
def sample(miner_id:str, env_id:str, challenge_id:str) -> Sample
def build_block(samples:list[Sample], prev_hash:str) -> Block
def verify_block(block:Block) -> tuple[bool, list[str]]        # ok, errors
def merge_and_score(blocks:list[Block]) -> dict                # winner, details
def set_winner_weights(winner_id:str) -> None
```

---

## 9) CLI (UX + exit codes)

```
af env run --env tictactoe-v0 --seed 123 --steps 9        # print obs/info/verdict locally
af duel --cont <uid> --champ <uid> --envs tictactoe-v0,mult8-v0 --eps 0.01 --K 1
af validate --cont <uid> --champ <uid> --push s3://bucket/prefix --block-size 100
af blocks pull --from https://.../index.json --to ./.cache/blocks
af merge --from ./.cache/blocks --print-winner
af set-weights --winner <uid> --commit-reveal
```

* Exit code `0` success, `1` verification failure, `2` network/storage error, `3` config error.

---

## 10) Configuration

* `affine.toml` (optional) or env vars:

  * `AFFINE_ENV_IDS = "tictactoe-v0,mult8-v0"`
  * `AFFINE_RATIO_EPS = 0.01`
  * `AFFINE_WILSON_Z = 1.96`
  * `AFFINE_ENV_N_CAP = 200`
  * `AFFINE_BLOCK_SIZE = 100`
  * `AFFINE_BUCKET_URL = s3://...`
  * `AFFINE_DECAY_HALFLIFE_DAYS = 7`
  * Timeouts: `AFFINE_REQ_TIMEOUT_S`, `AFFINE_MOVE_TIMEOUT_MS`

---

## 11) Observability

* **Structured logs** (JSON): per sample (`validator`, `env_id`, `challenge_id`, `uid`, `ok`, `latency_ms`, `bytes`), per block (hashes), per merge (winner, counts).
* **Metrics** (Prometheus):

  * `affine_samples_total{env,validator,ok_reason}`
  * `affine_duel_env_stops{env,winner}`
  * `affine_blocks_invalid_total{validator,reason}`
  * `affine_vtrust{validator}`
  * `affine_ratio_to_beat_env{env}`

---

## 12) Security & anti-cheat

* **No answer leakage:** prompts never embed truth; truth computed locally in `verify()`.
* **Timing & size signals:** record latencies and token counts; outliers flagged.
* **Duplicate suppression:** first `(env_id, challenge_id, miner_id)` counts; others logged, ignored, and penalize VTrust if systematic.
* **Block tamper-evidence:** broken prev_hash, signature mismatch, or merkle inconsistency ⇒ drop entire block and penalize VTrust.
* **Schedule discipline:** Only on-schedule challenge_ids (from commit→reveal) accepted.
* **Version pinning:** `spec_hash` change bumps `spec_version`; mixed versions in a block are allowed but must be declared in header.

---

## 13) Testing plan (deterministic, short)

### Unit / property

* **Env determinism:** For 10k seeds per env, `reset(seed)` produces identical `info/spec_hash`; `verify()` returns same verdict for same transcript.
* **Minimax correctness:** TTT judge matches an independent solver on random boards from seed.
* **Integer parse:** Fuzz `mult8.verify()` over messy outputs; 0 FP on non-integers; robust to whitespace/newlines.
* **Wilson harness:** Simulated Bernoulli streams at `p∈{0.45,0.5,0.55,0.7}`; acceptance/rejection rates meet 95% CI; mean `n` logged.

### Integration / e2e

* **Fake miners:** One perfect, one 90% win on TTT only → verify dethroning + ratio-to-beat update.
* **Blocks:** Tamper any field → verification fails; merkle proofs validate inclusion.
* **Merge:** Two validators with overlapping samples → same global winner; VTrust favors accurate one.

### Performance

* 1k samples/min/validator sustained locally; block write < 200 ms p95; merge 10k samples < 2 s.

---

## 14) Migration & rollout

1. **Branch `reboot/alpha`**; freeze main.
2. **Remove**: AgentGym, Pareto, round-robin, any env lacking deterministic `verify()`.
3. **Implement**: `envs` (ttt, mult8) + `verify()`.
4. **Implement**: `core.wilson`, `duel.local`, `duel.aggregate`, ratio decay.
5. **Implement**: validator blocks (`sampler`, `blocks`, `merge`, `vtrust`), chutes client, bittensor weights.
6. **Ship CLI** skeleton; wire minimal configs.
7. **Dry-run** (no weights on-chain) for 48h; compare independent validators; fix discrepancies.
8. **Enable set-weights** on testnet; monitor; then mainnet.

---

## 15) Developer checklists

### Module owners & deliverables

* **envs/**

  * `tictactoe.py`: deterministic start gen; minimax judge; action parsing; tests.
  * `mult8.py`: prompt builder; robust integer parse; tests.
  * `registry.py`: spec_versioning; spec_hash.
* **core/**

  * `rng.py`: PCG64 wrapper; tests.
  * `hashing.py`: canonical JSON; blake3/blake2s; merkle; tests.
  * `wilson.py`: CI calc; sequential helpers; property tests.
* **duel/**

  * `local.py`: streaming duel; budgets; timeouts; tests.
  * `aggregate.py`: multi-env early stop; ratio update; decay; tests.
* **validators/**

  * `sampler.py`: commit→reveal; schedule enforcement; sample builder; tests.
  * `blocks.py`: signing; verification; storage adapters; tests.
  * `merge.py`: dedupe; recompute verify; scoreboard; tests.
  * `vtrust.py`: metrics; penalties; tests.
  * `weights.py`: commit-reveal cadence; dry-run; tests.
* **miners/**

  * `chutes_client.py`: async query; request_id capture; retries; tests.
* **net/**

  * `bittensor.py`: minimal `set_weights`; mock for tests.

### Code review gates

* No network in envs; `verify()` is pure and re-entrant.
* Replaying any sample reproduces the verdict.
* Block and sample hashes stable across platforms.
* All CLI commands have helpful `--help` and sane exit codes.

---

## 16) Algorithms & formulas (for dev implementation)

### Wilson interval

For successes `w`, trials `n`, z-score `z`:

```
phat = w / n
den  = 1 + z*z/n
center = phat + z*z/(2*n)
margin = z * sqrt( (phat*(1-phat)/n) + (z*z)/(4*n*n) )
lo = (center - margin) / den
hi = (center + margin) / den
```

### Ratio-to-beat update (geometric mean)

* Collect per-env MLEs `r_e = wins_e / total_e` for envs the contender **won**.
* `g = exp( (1/|wins|) * sum(log(max(0.5, min(0.95, r_e)))) )`
* New `ratio_to_beat_env = min(0.95, max(0.5+ε, g))`
* Decay each epoch: `r ← 0.5 + (r − 0.5) * exp(-1/τ)`; default `τ` = half-life 7 days (or 14 epochs).

---

## 17) JSON examples (reference)

### Sample (mult8)

```json
{
  "env_id": "mult8-v0",
  "env_spec_version": 1,
  "challenge_id": "1f3a0c9d2b8e7c44aa0f6e1d0b4a9f02",
  "validator_hotkey": "val_abc",
  "miner_id": "uid_12",
  "role": "cont",
  "prompt": "Compute 12345678 × 87654321; return only the integer.",
  "response": "1082152022374638",
  "info": {"spec_hash": "b3:..."},
  "ok": true,
  "reason": "win",
  "request_id": "ch_req_...",
  "timing": {"latency_ms": 912},
  "bytes": {"prompt": 54, "response": 16},
  "hash": "b3:..."
}
```

### BlockHeader

```json
{
  "prev_hash": "b3:prev...",
  "block_index": 7,
  "ts_unix": 1760931123,
  "validator_hotkey": "val_abc",
  "env_spec_versions": {"tictactoe-v0":1,"mult8-v0":1},
  "sample_count": 100,
  "merkle_root": "b3:root...",
  "signature": "ed25519:..."
}
```

---

## 18) Performance budgets & limits

* Request timeout (chutes): 30 s; move timeout (TTT): 2 s; single-turn timeout (mult8): 10 s.
* Per-env cap `n_cap`: 200 default (configurable).
* Block size: 100 samples; block write < 250 ms p95.
* Validator CPU: 4 vCPU can sustain 1k samples/min across both envs.

---

## 19) Adding a new environment (playbook)

1. Create `envs/<name>.py` with `metadata.env_id`, `spec_version=1`.
2. Ensure **all** randomness is from `rng.py` seeded by `challenge_id`.
3. Implement `verify()` that recomputes truth and validates transcripts.
4. Add to `envs/registry.py` and expose `spec_hash`.
5. Provide 10 property tests: determinism, `verify()` agrees with ground truth on fuzzed replies.
6. Update `cli --envs` and default configs.

---

## 20) Runbook (ops)

* **Weights flip:** dry-run (`--dry`) first; if leader changes twice within a single epoch, hold last stable winner.
* **Disaster recovery:** if blocks corrupted, validators fall back to local cache; a new chain can be started by publishing a block with known `genesis_prev_hash = "0"*64`.
* **Kill-switch:** env-level feature flag to exclude a misbehaving env from aggregation.

---

That’s the whole spec. If you want, I can turn this into a GitHub issue checklist per file, or draft the JSON Schemas for `Sample`/`Block` and a short docstring pack you can paste into each module.
