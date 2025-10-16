# Affine Implementation Specification

## Executive Summary

This spec defines a complete rewrite of the Affine subnet codebase. The core principle is **determinism, verifiability, and minimalism**. Every evaluation is reproducible from a `challenge_id`. Every claim is externally auditable. Every line of code must justify its existence.

**Core changes:**
- Delete AgentGym, Pareto dominance, round-robin sampling
- Implement Gymnasium-based, procedurally-generated, deterministic environments
- Replace round-robin with **champion-vs-contender duels** using Wilson confidence intervals
- Independent validator sampling with hash-chained blocks and VTrust
- Winner-takes-all weights based on aggregated, verified samples

---

## 1. Repository Structure

```
affine/
├── core/
│   ├── types.py          # Dataclasses: Challenge, Sample, Verdict, Block
│   ├── rng.py            # Deterministic PRNG (PCG64) from challenge_id
│   ├── wilson.py         # Wilson score interval calculations
│   ├── hashing.py        # BLAKE3/SHA-256 utilities
│   └── judge.py          # Shared deterministic judging utilities
├── envs/
│   ├── base.py           # BaseEnv interface (Gymnasium)
│   ├── tictactoe.py      # Multi-turn: TicTacToe-v0
│   ├── mult8.py          # Single-turn: Mult8-v0
│   └── registry.py       # Environment factory (name → env)
├── duel/
│   ├── arena.py          # Single-env duel with Wilson stopping
│   └── aggregate.py      # Multi-env aggregation with early stopping
├── validator/
│   ├── sampler.py        # Independent sampling loop
│   ├── blocks.py         # Hash-chained block construction
│   ├── vtrust.py         # VTrust scoring from sample verification
│   └── weights.py        # Winner-takes-all weight computation
├── miner/
│   └── client.py         # Chutes query client
├── net/
│   ├── bittensor.py      # Bittensor subnet integration
│   └── chutes.py         # Chutes inference wrapper
├── cli.py                # CLI: validate, duel, set-weights
└── pyproject.toml
```

**Delete entirely:**
- All AgentGym code
- Pareto dominance logic
- Round-robin samplers
- Legacy evaluation code

---

## 2. Environments (Gymnasium-based, Deterministic, Verifiable)

### 2.1 Interface (`envs/base.py`)

All environments implement the Gymnasium `Env` interface with extensions:

```python
class BaseEnv(gym.Env):
    metadata = {
        'env_id': str,        # e.g., 'tictactoe-v0'
        'spec_version': int,  # Increment on breaking changes
    }
    
    def reset(self, seed: int | None = None, options=None) -> tuple[obs, info]:
        """
        Returns:
            obs: Environment observation
            info: {
                'challenge_id': str,      # Hex seed for reproducibility
                'difficulty': int,        # Optional difficulty level
                'spec_hash': str,         # Hash of env code + params
                'metadata': dict          # Additional verification data
            }
        """
        
    def step(self, action) -> tuple[obs, reward, terminated, truncated, info]:
        """Standard Gymnasium step."""
        
    def verify(self, transcript: list[dict], info: dict) -> Verdict:
        """
        Deterministic verification from transcript.
        
        Args:
            transcript: List of {role, content, action} dicts
            info: Original reset() info dict
            
        Returns:
            Verdict(ok: bool, reason: str)
        """
```

### 2.2 Challenge ID Generation

```python
# core/rng.py
def challenge_id_to_seed(challenge_id: str, env_id: str, spec_version: int) -> int:
    """
    Deterministic seed derivation.
    
    challenge_id: 32-byte hex string (256 bits)
    Returns: uint64 seed for np.random.Generator(PCG64)
    """
    data = f"{env_id}:{spec_version}:{challenge_id}".encode()
    hash_bytes = blake3(data).digest()[:8]
    return int.from_bytes(hash_bytes, 'little')
```

**Generation (validator-side):**
```python
def generate_challenge_id(validator_hotkey: str, env_id: str, counter: int, 
                          epoch_anchor: str) -> str:
    """
    epoch_anchor: Hash of recent Bittensor block (prevents pre-computation)
    """
    data = f"{validator_hotkey}:{env_id}:{counter}:{epoch_anchor}".encode()
    return blake3(data).hexdigest()
```

### 2.3 TicTacToe-v0 (`envs/tictactoe.py`)

**Specification:**
- **Observation:** 3×3 board state (flat array, 0=empty, 1=X, 2=O) + current player
- **Action:** `Discrete(9)` (0-8 for board positions)
- **Reward:** +1 win, 0 draw, -1 loss (for current player)
- **Determinism:** Board generation and opponent moves fully deterministic from seed
- **Opponent:** Perfect minimax (implementation ~80 lines)

**Generation:**
```python
def reset(self, seed=None, options=None):
    super().reset(seed=seed)
    self.rng = np.random.Generator(np.random.PCG64(seed))
    
    # Generate deterministic starting position
    # Options: empty board, or random legal position
    board = self._generate_position(self.rng)
    challenge_id = f"{seed:064x}"
    
    return self._get_obs(board), {
        'challenge_id': challenge_id,
        'spec_hash': self._spec_hash(),
        'starting_player': self.current_player
    }
```

**Verification:**
```python
def verify(self, transcript, info):
    # Reconstruct game from challenge_id
    seed = int(info['challenge_id'], 16)
    board = self._generate_position(np.random.Generator(np.random.PCG64(seed)))
    
    # Replay moves
    for step in transcript:
        if step['role'] == 'miner':
            action = step['action']
            if not self._is_valid(board, action):
                return Verdict(False, f"Illegal move: {action}")
            board = self._apply_move(board, action)
            
    # Check final state
    result = self._get_result(board)
    expected = transcript[-1]['reward']
    
    return Verdict(result == expected, f"Result mismatch: {result} vs {expected}")
```

### 2.4 Mult8-v0 (`envs/mult8.py`)

**Specification:**
- **Single-turn:** Compute `A × B` where A, B are 8-digit integers
- **Observation:** String prompt `"Compute {A} × {B}"`
- **Action:** String response (extract integer)
- **Reward:** 1 if correct, 0 otherwise

**Generation:**
```python
def reset(self, seed=None, options=None):
    super().reset(seed=seed)
    rng = np.random.Generator(np.random.PCG64(seed))
    
    # Generate 8-digit numbers (no leading zeros)
    a = rng.integers(10_000_000, 100_000_000)
    b = rng.integers(10_000_000, 100_000_000)
    self.ground_truth = a * b
    
    prompt = f"Compute {a} × {b}. Return only the integer result."
    challenge_id = f"{seed:064x}"
    
    return prompt, {
        'challenge_id': challenge_id,
        'ground_truth_hash': blake3(str(self.ground_truth).encode()).hexdigest()
    }
```

**Verification:**
```python
def verify(self, transcript, info):
    # Extract final integer from response
    response = transcript[-1]['content']
    extracted = self._extract_integer(response)
    
    # Recompute ground truth
    seed = int(info['challenge_id'], 16)
    rng = np.random.Generator(np.random.PCG64(seed))
    a = rng.integers(10_000_000, 100_000_000)
    b = rng.integers(10_000_000, 100_000_000)
    expected = a * b
    
    return Verdict(extracted == expected, 
                   f"Expected {expected}, got {extracted}")

def _extract_integer(self, text: str) -> int | None:
    """Extract last integer sequence from text."""
    import re
    matches = re.findall(r'-?\d+', text.replace(',', ''))
    return int(matches[-1]) if matches else None
```

---

## 3. Duel System (Champion vs Contender)

### 3.1 Single-Environment Duel (`duel/arena.py`)

**Wilson Score Interval:**
```python
# core/wilson.py
def wilson_interval(wins: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """
    Returns (lower_bound, upper_bound) for binomial proportion.
    
    Uses Wilson score interval for small-sample robustness.
    """
    if total == 0:
        return (0.0, 1.0)
        
    from scipy.stats import norm
    z = norm.ppf((1 + confidence) / 2)
    p_hat = wins / total
    
    denominator = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt((p_hat * (1 - p_hat) / total + z**2 / (4 * total**2))) / denominator
    
    return (max(0, center - margin), min(1, center + margin))
```

**Sequential Stopping:**
```python
# duel/arena.py
@dataclass
class DuelConfig:
    ratio_to_beat: float = 0.51  # Contender must beat this win rate
    confidence: float = 0.95
    max_samples: int = 5000
    min_samples: int = 30

def duel_single_env(
    env_id: str,
    contender_uid: int,
    champion_uid: int,
    config: DuelConfig
) -> DuelResult:
    """
    Run sequential duel on one environment.
    
    Returns when either:
    - Contender's lower bound > ratio_to_beat (contender wins)
    - Contender's upper bound < ratio_to_beat (champion wins)
    - max_samples reached (inconclusive)
    """
    wins = total = 0
    samples = []
    
    for i in range(config.max_samples):
        # Generate challenge
        challenge_id = generate_challenge_id(validator_hotkey, env_id, i, epoch_anchor)
        
        # Query both miners
        cont_sample = query_miner(contender_uid, env_id, challenge_id)
        champ_sample = query_miner(champion_uid, env_id, challenge_id)
        
        # Score (handle ties)
        cont_win = score_sample(cont_sample)
        champ_win = score_sample(champ_sample)
        
        if cont_win and not champ_win:
            wins += 1
            total += 1
        elif champ_win and not cont_win:
            total += 1
        # Else: tie, don't count
        
        samples.extend([cont_sample, champ_sample])
        
        # Check stopping condition
        if total >= config.min_samples:
            lower, upper = wilson_interval(wins, total, config.confidence)
            
            if lower > config.ratio_to_beat:
                return DuelResult('contender', wins, total, lower, upper, samples)
            if upper < config.ratio_to_beat:
                return DuelResult('champion', wins, total, lower, upper, samples)
    
    # Max samples reached
    lower, upper = wilson_interval(wins, total, config.confidence)
    return DuelResult('inconclusive', wins, total, lower, upper, samples)
```

### 3.2 Multi-Environment Aggregation (`duel/aggregate.py`)

```python
@dataclass
class AggregateConfig:
    envs: list[str]           # ['tictactoe-v0', 'mult8-v0']
    K: int = 1                # Margin: need (N+K)/(2N) wins
    duel_config: DuelConfig

def duel_aggregate(
    contender_uid: int,
    champion_uid: int,
    config: AggregateConfig
) -> AggregateResult:
    """
    Run duels across environments with early stopping.
    
    Returns when:
    - Contender wins >= (N+K)/(2N) envs → contender wins
    - Contender cannot reach (N+K)/(2N) → champion wins
    """
    N = len(config.envs)
    threshold = (N + config.K) / (2 * N)
    
    env_results = {}
    cont_wins = 0
    completed = 0
    
    for env_id in config.envs:
        result = duel_single_env(env_id, contender_uid, champion_uid, config.duel_config)
        env_results[env_id] = result
        completed += 1
        
        if result.winner == 'contender':
            cont_wins += 1
            
        # Early stopping checks
        if cont_wins / N >= threshold:
            return AggregateResult('contender', env_results, cont_wins, N)
            
        remaining = N - completed
        max_possible = (cont_wins + remaining) / N
        if max_possible < threshold:
            return AggregateResult('champion', env_results, cont_wins, N)
    
    # All envs completed
    winner = 'contender' if cont_wins / N >= threshold else 'champion'
    return AggregateResult(winner, env_results, cont_wins, N)
```

### 3.3 Ratio-to-Beat Schedule

```python
# duel/aggregate.py
class RatioSchedule:
    def __init__(self, initial: float = 0.51, decay_halflife: int = 7):
        self.initial = initial
        self.current = initial
        self.decay_halflife = decay_halflife  # epochs
        self.last_update_epoch = 0
        
    def update_on_win(self, env_results: dict[str, DuelResult], current_epoch: int):
        """
        Update ratio to geometric mean of contender's win rates.
        """
        win_rates = []
        for result in env_results.values():
            if result.winner == 'contender' and result.total > 0:
                # Use MLE win rate from the duel
                rate = result.wins / result.total
                win_rates.append(rate)
        
        if win_rates:
            from scipy.stats import gmean
            new_ratio = min(0.95, gmean(win_rates))
            self.current = new_ratio
            self.last_update_epoch = current_epoch
            
    def decay(self, current_epoch: int):
        """
        Exponential decay toward initial value.
        """
        epochs_since = current_epoch - self.last_update_epoch
        decay_factor = 0.5 ** (epochs_since / self.decay_halflife)
        self.current = self.initial + (self.current - self.initial) * decay_factor
```

---

## 4. Validator System

### 4.1 Sample Schema

```python
# core/types.py
@dataclass(frozen=True)
class Sample:
    # Identity
    challenge_id: str
    env_id: str
    env_spec_version: int
    miner_uid: int
    role: str  # 'contender' or 'champion'
    
    # Invocation
    chute_invocation_id: str | None
    timestamp: int
    
    # Transcript
    prompt: str
    response: str
    transcript: list[dict]  # Full step-by-step if multi-turn
    
    # Verdict
    verdict: Verdict
    
    # Metadata
    latency_ms: int
    tokens_used: int | None
    
    def hash(self) -> str:
        """BLAKE3 hash of canonical JSON representation."""
        canonical = json.dumps(asdict(self), sort_keys=True)
        return blake3(canonical.encode()).hexdigest()
```

### 4.2 Block Chain (`validator/blocks.py`)

```python
@dataclass(frozen=True)
class BlockHeader:
    prev_hash: str
    block_index: int
    timestamp: int
    validator_hotkey: str
    
    env_spec_versions: dict[str, int]  # {env_id: version}
    sample_count: int
    merkle_root: str
    
    signature: str  # Ed25519 signature of header fields

@dataclass(frozen=True)
class Block:
    header: BlockHeader
    sample_hashes: list[str]
    samples: list[Sample]  # Optional: can store separately
    
    def hash(self) -> str:
        """Hash of header + sample_hashes."""
        data = asdict(self.header)
        data['sample_hashes'] = self.sample_hashes
        canonical = json.dumps(data, sort_keys=True)
        return blake3(canonical.encode()).hexdigest()
        
    def verify_signature(self, public_key: bytes) -> bool:
        """Verify header signature."""
        # Implement Ed25519 verification
        
    def verify_merkle_root(self) -> bool:
        """Verify merkle_root matches sample_hashes."""
        computed = compute_merkle_root(self.sample_hashes)
        return computed == self.header.merkle_root
```

**Block Construction:**
```python
# validator/blocks.py
def build_block(
    samples: list[Sample],
    prev_hash: str,
    validator_hotkey: str,
    signing_key: bytes
) -> Block:
    """
    Build and sign a block of samples.
    """
    sample_hashes = [s.hash() for s in samples]
    merkle_root = compute_merkle_root(sample_hashes)
    
    env_versions = {s.env_id: s.env_spec_version for s in samples}
    
    header = BlockHeader(
        prev_hash=prev_hash,
        block_index=get_next_index(),
        timestamp=int(time.time()),
        validator_hotkey=validator_hotkey,
        env_spec_versions=env_versions,
        sample_count=len(samples),
        merkle_root=merkle_root,
        signature=""  # Fill after hashing
    )
    
    # Sign header
    header_data = asdict(header)
    del header_data['signature']
    signature = sign_ed25519(signing_key, json.dumps(header_data, sort_keys=True))
    header = replace(header, signature=signature)
    
    return Block(header, sample_hashes, samples)
```

### 4.3 VTrust Computation (`validator/vtrust.py`)

```python
def compute_vtrust(
    validator_blocks: list[Block],
    global_samples: dict[str, Sample]  # challenge_id → canonical sample
) -> dict[str, float]:
    """
    Compute VTrust for each validator based on overlap accuracy.
    
    Returns: {validator_hotkey: vtrust_score}
    """
    vtrust = {}
    
    for validator_hotkey, blocks in group_by_validator(validator_blocks):
        correct = 0
        total = 0
        
        for block in blocks:
            for sample in block.samples:
                cid = sample.challenge_id
                
                if cid in global_samples:
                    canonical = global_samples[cid]
                    
                    # Verify this validator's verdict matches canonical
                    if sample.verdict == canonical.verdict:
                        correct += 1
                    total += 1
                else:
                    # Sample not in global set, penalize
                    total += 1
        
        # Wilson lower bound of correct/total
        if total > 0:
            lower, _ = wilson_interval(correct, total, confidence=0.95)
            vtrust[validator_hotkey] = lower
        else:
            vtrust[validator_hotkey] = 0.0
            
    return vtrust
```

**Canonical Sample Resolution:**
```python
def build_canonical_samples(
    all_blocks: list[Block],
    vtrust: dict[str, float]
) -> dict[str, Sample]:
    """
    Build global canonical sample set weighted by VTrust.
    
    For each challenge_id:
    - Group samples from different validators
    - Weight verdicts by validator VTrust
    - Take majority verdict as canonical
    """
    samples_by_challenge = defaultdict(list)
    
    for block in all_blocks:
        validator = block.header.validator_hotkey
        for sample in block.samples:
            samples_by_challenge[sample.challenge_id].append(
                (validator, sample)
            )
    
    canonical = {}
    for cid, validator_samples in samples_by_challenge.items():
        # Weight verdicts by VTrust
        votes = defaultdict(float)
        for validator, sample in validator_samples:
            weight = vtrust.get(validator, 0.0)
            votes[sample.verdict] += weight
            
        # Majority verdict
        winning_verdict = max(votes.items(), key=lambda x: x[1])[0]
        
        # Use sample with highest VTrust validator
        canonical_sample = max(
            validator_samples,
            key=lambda x: vtrust.get(x[0], 0.0)
        )[1]
        
        canonical[cid] = replace(canonical_sample, verdict=winning_verdict)
        
    return canonical
```

### 4.4 Weight Setting (`validator/weights.py`)

```python
def compute_weights(
    canonical_samples: dict[str, Sample],
    current_champion_uid: int,
    all_miner_uids: list[int]
) -> dict[int, float]:
    """
    Compute winner-takes-all weights from canonical samples.
    
    Returns: {miner_uid: weight}
    """
    # Group samples by miner
    miner_scores = defaultdict(lambda: {'wins': 0, 'total': 0})
    
    for sample in canonical_samples.values():
        miner = sample.miner_uid
        if sample.verdict.ok:
            miner_scores[miner]['wins'] += 1
        miner_scores[miner]['total'] += 1
    
    # Compute win rates
    win_rates = {
        miner: scores['wins'] / scores['total'] if scores['total'] > 0 else 0.0
        for miner, scores in miner_scores.items()
    }
    
    # Winner-takes-all
    if win_rates:
        winner_uid = max(win_rates.items(), key=lambda x: x[1])[0]
    else:
        winner_uid = current_champion_uid
    
    weights = {uid: 0.0 for uid in all_miner_uids}
    weights[winner_uid] = 1.0
    
    return weights
```

---

## 5. Anti-Exploit Measures

### 5.1 Deterministic Challenge Generation

```python
# validator/sampler.py
class ChallengeCommitment:
    """Commit-reveal for challenge generation."""
    
    def __init__(self, validator_hotkey: str):
        self.validator_hotkey = validator_hotkey
        self.commitments = {}  # epoch → commitment_hash
        
    def commit(self, epoch: int, seed: bytes) -> str:
        """Commit to a seed for an epoch."""
        commitment = blake3(seed).hexdigest()
        self.commitments[epoch] = commitment
        return commitment
        
    def reveal(self, epoch: int, seed: bytes) -> bool:
        """Reveal seed and verify commitment."""
        expected = self.commitments.get(epoch)
        if not expected:
            return False
        actual = blake3(seed).hexdigest()
        return actual == expected
        
    def generate_challenges(
        self, 
        epoch: int, 
        seed: bytes, 
        env_id: str, 
        count: int
    ) -> list[str]:
        """Generate challenge_ids from revealed seed."""
        if not self.reveal(epoch, seed):
            raise ValueError("Invalid seed reveal")
            
        challenges = []
        for i in range(count):
            data = f"{seed.hex()}:{env_id}:{i}".encode()
            challenge_id = blake3(data).hexdigest()
            challenges.append(challenge_id)
            
        return challenges
```

### 5.2 Duplicate Detection

```python
# validator/sampler.py
class DuplicateDetector:
    """Prevent duplicate challenge evaluation."""
    
    def __init__(self, window_size: int = 10000):
        self.seen = {}  # (env_id, challenge_id, miner_uid) → timestamp
        self.window_size = window_size
        
    def check_and_mark(self, env_id: str, challenge_id: str, miner_uid: int) -> bool:
        """
        Returns True if this is a duplicate.
        Marks as seen if novel.
        """
        key = (env_id, challenge_id, miner_uid)
        
        if key in self.seen:
            return True  # Duplicate
            
        # Mark as seen
        self.seen[key] = time.time()
        
        # Prune old entries
        if len(self.seen) > self.window_size:
            self._prune_old()
            
        return False
        
    def _prune_old(self):
        """Keep only most recent window_size entries."""
        sorted_items = sorted(self.seen.items(), key=lambda x: x[1])
        self.seen = dict(sorted_items[-self.window_size:])
```

### 5.3 Transcript Validation

```python
# core/judge.py
def validate_transcript(transcript: list[dict], max_steps: int = 1000) -> bool:
    """
    Validate transcript structure and content.
    """
    if len(transcript) > max_steps:
        return False
        
    for i, step in enumerate(transcript):
        # Required fields
        if not all(k in step for k in ['role', 'content']):
            return False
            
        # Role must be valid
        if step['role'] not in ['env', 'miner', 'minerA', 'minerB']:
            return False
            
        # Content must be string
        if not isinstance(step['content'], str):
            return False
            
        # Reasonable content length
        if len(step['content']) > 100_000:  # 100KB per step
            return False
            
    return True
```

---

## 6. Implementation Priorities

### Phase 1: Core Infrastructure (Week 1)
1. `core/types.py` - Data structures
2. `core/rng.py` - Deterministic RNG
3. `core/wilson.py` - Statistical tests
4. `core/hashing.py` - Cryptographic primitives

### Phase 2: Environments (Week 2)
1. `envs/base.py` - Gymnasium interface
2. `envs/mult8.py` - Single-turn environment
3. `envs/tictactoe.py` - Multi-turn environment
4. `envs/registry.py` - Environment factory

### Phase 3: Duel System (Week 3)
1. `duel/arena.py` - Single-env sequential testing
2. `duel/aggregate.py` - Multi-env aggregation
3. Ratio-to-beat schedule implementation

### Phase 4: Validator System (Week 4)
1. `validator/blocks.py` - Block chain construction
2. `validator/sampler.py` - Independent sampling
3. `validator/vtrust.py` - Trust computation
4. `validator/weights.py` - Weight setting

### Phase 5: Integration (Week 5)
1. `net/bittensor.py` - Subnet integration
2. `net/chutes.py` - Miner query client
3. `cli.py` - Command-line interface
4. End-to-end testing

---

## 7. Testing Strategy

### Unit Tests
```python
# tests/test_wilson.py
def test_wilson_interval_properties():
    """Wilson intervals should contain true p with high probability."""
    for true_p in [0.3, 0.5, 0.7]:
        for n in [10, 50, 100, 500]:
            # Simulate binomial trials
            successes = np.random.binomial(n, true_p, 1000)
            
            # Check coverage
            intervals = [wilson_interval(s, n) for s in successes]
            coverage = sum(lo <= true_p <= hi for lo, hi in intervals) / len(intervals)
            
            assert coverage >= 0.90  # 95% CI should have ~95% coverage

# tests/test_envs.py
def test_env_determinism():
    """Same seed must produce identical challenges."""
    env = registry.make('mult8-v0')
    
    seed = 12345
    obs1, info1 = env.reset(seed=seed)
    obs2, info2 = env.reset(seed=seed)
    
    assert obs1 == obs2
    assert info1 == info2
    assert info1['challenge_id'] == info2['challenge_id']

def test_env_verification():
    """Verification must be deterministic and match execution."""
    env = registry.make('tictactoe-v0')
    obs, info = env.reset(seed=12345)
    
    # Play a game
    transcript = []
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, step_info = env.step(action)
        transcript.append({'role': 'miner', 'action': action, 'content': str(action)})
        done = terminated or truncated
        
    # Verify
    verdict = env.verify(transcript, info)
    assert verdict.ok  # Should match actual game outcome
```

### Integration Tests
```python
# tests/test_duel.py
async def test_duel_convergence():
    """Duel should converge to correct winner."""
    # Mock miners: one with 60% win rate, one with 40%
    config = DuelConfig(ratio_to_beat=0.5, max_samples=1000)
    
    result = await duel_single_env('mult8-v0', contender_uid=1, champion_uid=2, config)
    
    # Should detect contender is better
    assert result.winner == 'contender'
    assert result.wins / result.total > 0.5

# tests/test_blocks.py
def test_block_integrity():
    """Block tampering should be detectable."""
    samples = [create_mock_sample() for _ in range(100)]
    block = build_block(samples, prev_hash="0" * 64, validator_hotkey="test", signing_key=key)
    
    # Verify original
    assert block.verify_signature(public_key)
    assert block.verify_merkle_root()
    
    # Tamper with sample
    tampered = replace(block, samples=samples[:-1])
    assert not tampered.verify_merkle_root()
```

---

## 8. Migration Plan

### Pre-Migration
1. Archive current codebase: `git tag legacy-v1`
2. Create migration branch: `git checkout -b reboot/v2`
3. Document current miners and validators

### Migration Steps
1. **Delete legacy code:**
   ```bash
   rm -rf affine/agentenvironments/
   rm -rf affine/protocol/pareto.py
   rm -rf affine/validator/roundrobin.py
   # ... (delete all AgentGym, Pareto, round-robin code)
   ```

2. **Implement new structure** (following Phase 1-5 above)

3. **Testnet deployment:**
   - Deploy to testnet with synthetic miners
   - Run validator for 1 week
   - Monitor convergence and VTrust

4. **Mainnet cutover:**
   - Announce migration 2 weeks in advance
   - Deploy to mainnet
   - Run validators in parallel for 1 epoch
   - Switch weights to new system

### Rollback Plan
- Keep legacy validator running in read-only mode for 1 month
- If critical issues detected, revert to legacy within 48 hours
- Document all differences between legacy and new verdicts

---

## 9. Configuration Defaults

```python
# config/defaults.py
WILSON_CONFIDENCE = 0.95
RATIO_TO_BEAT_INITIAL = 0.51
RATIO_TO_BEAT_DECAY_HALFLIFE = 7  # epochs
MIN_SAMPLES_PER_ENV = 30
MAX_SAMPLES_PER_ENV = 5000
EARLY_STOP_MARGIN_K = 1  # Need (N+1)/(2N) env wins
BLOCK_SIZE = 100  # samples per block
VTRUST_CONFIDENCE = 0.95
MAX_TRANSCRIPT_LENGTH = 1000  # steps
MAX_STEP_CONTENT_SIZE = 100_000  # bytes
DUPLICATE_WINDOW_SIZE = 10_000  # challenge_ids
```

---

## 10. Success Criteria

The implementation is complete when:

1. **Determinism:** 10,000 seeds produce identical challenges across runs
2. **Convergence:** Duel system correctly identifies better miner in <200 samples for 10% win-rate difference
3. **Integrity:** Block tampering detection rate >99.9%
4. **Efficiency:** Validator can process 1000 samples/hour on commodity hardware
5. **Robustness:** No exploits found in 2-week red-team exercise
6. **Code quality:** <5000 lines of core code (excluding tests)

---

## Appendix A: Key Algorithms (Pseudocode)

### Wilson Score Interval
```
function wilson_interval(wins, total, z=1.96):
    if total == 0: return (0, 1)
    
    p_hat = wins / total
    denominator = 1 + z² / total
    center = (p_hat + z² / (2 * total)) / denominator
    margin = z * sqrt((p_hat * (1 - p_hat) / total + z² / (4 * total²))) / denominator
    
    return (max(0, center - margin), min(1, center + margin))
```

### Sequential Duel
```
function duel(contender, champion, ratio_to_beat):
    wins = 0
    total = 0
    
    while total < MAX_SAMPLES:
        challenge = generate_challenge()
        
        cont_result = query(contender, challenge)
        champ_result = query(champion, challenge)
        
        if cont_result > champ_result:
            wins += 1
            total += 1
        elif champ_result > cont_result:
            total += 1
        # else: tie, skip
        
        if total >= MIN_SAMPLES:
            lower, upper = wilson_interval(wins, total)
            
            if lower > ratio_to_beat:
                return CONTENDER_WINS
            if upper < ratio_to_beat:
                return CHAMPION_WINS
    
    return INCONCLUSIVE
```
