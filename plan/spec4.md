# Affine v2 Implementation Specification

## Overview

Complete rewrite of Affine as a minimal, deterministic, verifiable Bittensor subnet. Core principle: **every evaluation must be reproducible from a challenge ID**.

## Architecture

### Directory Structure

```
affine/
├── core/
│   ├── types.py          # Challenge, Sample, Verdict, Block dataclasses
│   ├── rng.py            # Deterministic RNG from challenge_id
│   ├── wilson.py         # Wilson score interval calculations
│   └── hashing.py        # Block hashing and verification
├── envs/
│   ├── base.py           # BaseEnv abstract class with Gym interface
│   ├── tictactoe.py      # Multi-turn tic-tac-toe environment
│   ├── multiply.py       # Single-turn 8-digit multiplication
│   └── registry.py       # Environment registration and versioning
├── evaluation/
│   ├── duel.py           # Single-env contender vs champion evaluation
│   ├── aggregator.py     # Multi-env aggregation with early stopping
│   └── ratio.py          # Ratio-to-beat tracking and decay
├── validator/
│   ├── sampler.py        # Independent sampling loop
│   ├── blocks.py         # Block creation and hash chaining
│   ├── vtrust.py         # VTrust calculation from sample accuracy
│   └── weights.py        # Winner-takes-all weight computation
├── miner/
│   └── client.py         # Chutes API client
├── storage/
│   └── bucket.py         # Shared bucket interface for blocks
└── cli.py                # CLI entry points
```

## Core Components

### 1. Environment System

#### Base Environment (`envs/base.py`)

```python
class BaseEnv(gym.Env):
    """
    Abstract base for all Affine environments.
    Enforces deterministic generation and verification.
    """
    
    @property
    @abstractmethod
    def env_id(self) -> str:
        """Unique environment identifier with version"""
        
    @abstractmethod
    def generate_challenge(self, challenge_id: str) -> dict:
        """Generate challenge from deterministic seed"""
        
    @abstractmethod
    def verify(self, response: str, challenge_data: dict) -> Verdict:
        """Verify response against ground truth"""
        
    def reset(self, seed: int = None) -> tuple[Any, dict]:
        """Reset with challenge_id in info dict"""
        if seed is None:
            seed = self._generate_seed()
        self.rng = np.random.Generator(np.random.PCG64(seed))
        challenge_id = f"{self.env_id}:{seed:016x}"
        self.challenge_data = self.generate_challenge(challenge_id)
        
        obs = self._get_observation()
        info = {
            "challenge_id": challenge_id,
            "env_id": self.env_id,
            "metadata": self.challenge_data.get("metadata", {})
        }
        return obs, info
```

#### Tic-Tac-Toe Environment (`envs/tictactoe.py`)

- **Multi-turn** game between two miners
- **Deterministic** starting positions from challenge_id
- **Observation**: 3x3 board state as flat array
- **Action**: Position index (0-8)
- **Verification**: Replay game from transcript, verify legal moves and outcome

#### Multiplication Environment (`envs/multiply.py`)

- **Single-turn** computation task
- **Deterministic** 8-digit number pairs from challenge_id
- **Observation**: Two numbers as strings
- **Action**: Product as string
- **Verification**: Exact match of integer product

### 2. Evaluation System

#### Duel Engine (`evaluation/duel.py`)

```python
class DuelEngine:
    """
    Evaluates contender vs champion on single environment.
    Uses Wilson score interval for statistical confidence.
    """
    
    def evaluate_env(
        self,
        env: BaseEnv,
        contender: str,
        champion: str,
        confidence: float = 0.95,
        max_samples: int = 1000
    ) -> DuelResult:
        """
        Run sequential evaluation until confident winner.
        Returns: DuelResult with winner, confidence, samples
        """
        wins_contender = 0
        total_samples = 0
        
        while total_samples < max_samples:
            # Generate challenge
            obs, info = env.reset()
            challenge_id = info["challenge_id"]
            
            # Get responses
            response_cont = self.query_miner(contender, obs, challenge_id)
            response_champ = self.query_miner(champion, obs, challenge_id)
            
            # Verify responses
            verdict_cont = env.verify(response_cont, env.challenge_data)
            verdict_champ = env.verify(response_champ, env.challenge_data)
            
            # Update counts
            if verdict_cont.success and not verdict_champ.success:
                wins_contender += 1
            total_samples += 1
            
            # Check Wilson bounds
            lower, upper = wilson_score_interval(
                wins_contender, total_samples, confidence
            )
            
            # Early stopping conditions
            if lower > 0.5:  # Contender confidently better
                return DuelResult(winner="contender", ...)
            if upper < 0.5:  # Champion confidently better
                return DuelResult(winner="champion", ...)
                
        return DuelResult(winner="inconclusive", ...)
```

#### Multi-Environment Aggregator (`evaluation/aggregator.py`)

```python
class Aggregator:
    """
    Aggregates results across multiple environments.
    Implements early stopping when outcome is determined.
    """
    
    def evaluate_all(
        self,
        envs: list[BaseEnv],
        contender: str,
        champion: str,
        required_wins: float = 0.5  # Fraction of envs to win
    ) -> AggregateResult:
        """
        Evaluate across all environments with early stopping.
        """
        total_envs = len(envs)
        target_wins = math.ceil(total_envs * required_wins)
        
        wins = 0
        losses = 0
        
        for env in envs:
            # Can contender still win?
            remaining = total_envs - wins - losses
            if wins + remaining < target_wins:
                return AggregateResult(winner="champion", ...)
                
            # Can champion still defend?
            if losses + remaining < (total_envs - target_wins + 1):
                return AggregateResult(winner="contender", ...)
                
            # Run duel on this environment
            result = self.duel_engine.evaluate_env(env, contender, champion)
            
            if result.winner == "contender":
                wins += 1
            elif result.winner == "champion":
                losses += 1
                
        # Final determination
        if wins >= target_wins:
            return AggregateResult(winner="contender", ...)
        else:
            return AggregateResult(winner="champion", ...)
```

#### Ratio Management (`evaluation/ratio.py`)

```python
class RatioManager:
    """
    Manages the dynamic "ratio to beat" threshold.
    """
    
    def __init__(self, initial_ratio: float = 0.51, decay_rate: float = 0.01):
        self.current_ratio = initial_ratio
        self.decay_rate = decay_rate
        self.last_update = time.time()
        
    def update_on_victory(self, win_ratios: list[float]):
        """Update ratio to geometric mean of winner's performance"""
        geometric_mean = np.exp(np.mean(np.log(win_ratios)))
        self.current_ratio = min(geometric_mean, 0.95)  # Cap at 95%
        self.last_update = time.time()
        
    def get_current_ratio(self) -> float:
        """Get ratio with exponential decay applied"""
        elapsed = time.time() - self.last_update
        decayed = 0.5 + (self.current_ratio - 0.5) * np.exp(-self.decay_rate * elapsed)
        return max(decayed, 0.51)  # Floor at 51%
```

### 3. Validator System

#### Independent Sampler (`validator/sampler.py`)

```python
class Sampler:
    """
    Generates challenges and collects miner responses.
    """
    
    def sample_miners(
        self,
        miners: list[str],
        env: BaseEnv,
        num_samples: int
    ) -> list[Sample]:
        """
        Generate challenges and query miners independently.
        """
        samples = []
        
        for _ in range(num_samples):
            obs, info = env.reset()
            challenge_id = info["challenge_id"]
            
            for miner in miners:
                # Query via Chutes
                response, invocation_id = self.query_chutes(
                    miner, obs, challenge_id
                )
                
                # Verify response
                verdict = env.verify(response, env.challenge_data)
                
                # Create sample
                sample = Sample(
                    challenge_id=challenge_id,
                    env_id=env.env_id,
                    miner_id=miner,
                    invocation_id=invocation_id,
                    prompt=self.format_prompt(obs),
                    response=response,
                    verdict=verdict,
                    timestamp=time.time()
                )
                samples.append(sample)
                
        return samples
```

#### Block Chain (`validator/blocks.py`)

```python
class BlockChain:
    """
    Creates hash-chained blocks of samples.
    """
    
    def create_block(
        self,
        samples: list[Sample],
        prev_hash: str,
        validator_id: str
    ) -> Block:
        """
        Create a new block with samples.
        """
        # Hash all samples
        sample_hashes = [self.hash_sample(s) for s in samples]
        merkle_root = self.compute_merkle_root(sample_hashes)
        
        block = Block(
            prev_hash=prev_hash,
            block_height=self.get_next_height(),
            timestamp=time.time(),
            validator_id=validator_id,
            samples=samples,
            sample_hashes=sample_hashes,
            merkle_root=merkle_root
        )
        
        # Sign block
        block.signature = self.sign_block(block)
        block.hash = self.hash_block(block)
        
        return block
```

#### VTrust Calculator (`validator/vtrust.py`)

```python
class VTrustCalculator:
    """
    Calculates validator trust from sample accuracy.
    """
    
    def calculate_vtrust(
        self,
        validator_blocks: dict[str, list[Block]]
    ) -> dict[str, float]:
        """
        Calculate VTrust scores for all validators.
        """
        vtrust_scores = {}
        
        for validator_id, blocks in validator_blocks.items():
            correct = 0
            total = 0
            
            for block in blocks:
                for sample in block.samples:
                    # Recreate challenge
                    env = self.registry.get_env(sample.env_id)
                    obs, info = env.reset(seed=self.parse_seed(sample.challenge_id))
                    
                    # Verify verdict
                    true_verdict = env.verify(sample.response, env.challenge_data)
                    
                    if true_verdict.success == sample.verdict.success:
                        correct += 1
                    total += 1
                    
            # Wilson score for trust
            if total > 0:
                lower, _ = wilson_score_interval(correct, total, 0.95)
                vtrust_scores[validator_id] = lower
            else:
                vtrust_scores[validator_id] = 0.0
                
        return vtrust_scores
```

#### Weight Setter (`validator/weights.py`)

```python
class WeightSetter:
    """
    Computes and sets winner-takes-all weights.
    """
    
    def compute_weights(
        self,
        all_blocks: list[Block],
        vtrust_scores: dict[str, float]
    ) -> dict[str, float]:
        """
        Aggregate samples and determine winner.
        """
        # Aggregate samples weighted by VTrust
        weighted_wins = defaultdict(float)
        weighted_total = defaultdict(float)
        
        for block in all_blocks:
            validator_trust = vtrust_scores.get(block.validator_id, 0)
            
            for sample in block.samples:
                weight = validator_trust
                weighted_total[sample.miner_id] += weight
                if sample.verdict.success:
                    weighted_wins[sample.miner_id] += weight
                    
        # Find best performer
        best_miner = None
        best_ratio = 0
        
        for miner_id in weighted_total:
            if weighted_total[miner_id] > 0:
                ratio = weighted_wins[miner_id] / weighted_total[miner_id]
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_miner = miner_id
                    
        # Winner takes all
        weights = {miner: 0.0 for miner in weighted_total}
        if best_miner:
            weights[best_miner] = 1.0
            
        return weights
```

### 4. Storage System

#### Shared Bucket (`storage/bucket.py`)

```python
class SharedBucket:
    """
    Interface for storing and retrieving blocks.
    """
    
    def push_block(self, block: Block) -> str:
        """Upload block and return URL"""
        
    def pull_blocks(self, validator_id: str = None) -> list[Block]:
        """Retrieve blocks, optionally filtered by validator"""
        
    def verify_chain(self, blocks: list[Block]) -> bool:
        """Verify hash chain integrity"""
```

## Implementation Guidelines

### Principles

1. **Determinism First**: Every challenge must be reproducible from its ID
2. **Minimal Dependencies**: Use stdlib where possible, avoid heavy frameworks
3. **Pure Functions**: Verification and scoring must be side-effect free
4. **Fail Fast**: Invalid inputs should error immediately with clear messages
5. **No Hidden State**: All randomness comes from explicit seeds

### Security Considerations

- Challenge IDs include validator signature to prevent prediction
- Blocks are hash-chained to prevent tampering
- VTrust penalizes incorrect verifications
- All transcripts are stored for auditability
- Time limits prevent DoS attacks

### Performance Targets

- Single challenge evaluation: < 100ms (excluding miner query)
- Block verification: < 10ms per sample
- Wilson confidence convergence: < 100 samples typical
- Storage overhead: ~1KB per sample

## Migration Path

1. **Phase 1**: Delete all existing code except Bittensor interface
2. **Phase 2**: Implement core types and environments
3. **Phase 3**: Implement evaluation system
4. **Phase 4**: Implement validator system
5. **Phase 5**: Integration testing on testnet
6. **Phase 6**: Deploy to mainnet with backward compatibility period

## Testing Requirements

- Environment determinism: 10,000 seeds → identical challenges
- Wilson convergence: Property tests on known distributions
- Block integrity: Tamper detection tests
- End-to-end: Simulated network with multiple validators
