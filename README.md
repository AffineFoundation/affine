# Affine

## Prerequisites
1. **Register on Chutes**: [https://github.com/rayonlabs/chutes](https://github.com/rayonlabs/chutes) - Get your API keys
2. **Enable Developer Account**: [https://chutes.ai/app/docs](https://chutes.ai/app/docs) - Required for deployment


## Installation
```bash
# Install uv Astral
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install Affine
git clone https://github.com/AffineFoundation/affine.git
cd affine
uv venv && source .venv/bin/activate
uv pip install -e .

# Verify installation
af --help
```


## Quick Start
```bash
# Set your chutes api key
uv run af set CHUTES_API_KEY <your_key_value>
uv run af set HF_TOKEN <your_HF_TOKEN>

# Generate sample challenges for evaluation (recommended)
uv run af gen

# Eval uid:5 miner on the SAT env with 10 samples
uv run af -vv run 5 SAT -n 10

# Deploy models
af deploy /path/to/model
```

## Sample Management

Affine includes a sophisticated sample management system that pre-generates challenges for efficient evaluation:

### Generate Samples
```bash
# Generate samples for all environments (maintains stock of 100 per env)
uv run af gen

# Generate samples for specific environment
uv run af gen coin -n 50

# Generate with custom stock target
uv run af gen --stock 200
```

### View Sample Statistics
```bash
# View statistics for all environments
uv run af samples

# View statistics for specific environment
uv run af samples sat
```

### Using Samples in Evaluation
```bash
# All challenges are sourced from pre-generated samples (samples are consumed and deleted)
uv run af run 5 SAT -n 10
```

## Key Features

- **Always Use Samples**: Challenges are always sourced from pre-generated JSONL files
- **Auto-Generation**: Missing samples are generated and saved automatically
- **One-time Use**: Samples are deleted after use to prevent reuse
- **Background Replenishment**: Stock is automatically replenished in the background
- **Smart Stock Management**: Maintains target inventory levels automatically

## Validation & Scoring

Affine includes a comprehensive validation system using GRPO (Group Relative Performance Optimization) that tests miners across multiple domains with Sybil-proof scoring:

### Basic Validation
```bash
# Validate specific miners with default settings (10 samples per environment, all environments)
uv run af validate --uids "1,2,3,4,5"

# Validate with specific environment and custom samples
uv run af validate --uids "1,2,3,4,5" --env "SAT" --samples 20

# Validate with multiple environments
uv run af validate --uids "1,2,3,4,5" --env "SAT,ABD" --samples 15

# Multiple validation cycles
uv run af validate --uids "1,2,3,4,5" --cycles 3

# Continuous validation (runs until interrupted)
uv run af validate --uids "1,2,3,4,5" --cycles 0

# Validate all available miners across all environments
uv run af validate --env "ALL" --samples 15
```

### Check Evaluation Scores
```bash
# View eval scores for specific miners
uv run af scores --uids "1,2,3,4,5"

# View all eval scores (top 20)
uv run af scores

# View more miners
uv run af scores --limit 50
```

### Validation Options

- **`--uids`**: Specify which miners to validate (comma-separated)
- **`--env`**: Choose environment(s) to test (SAT/ABD/COIN/ALL, or comma-separated like "SAT,ABD", default: ALL)
- **`--samples`**: Number of samples each miner faces per environment (default: 10)
- **`--cycles`**: Number of validation cycles (default: 1, use 0 for continuous)
- **`--ema-alpha`**: Exponential moving average smoothing factor (default: 0.1)
- **`--skew-penalty`**: Weight for domain skew penalty (default: 0.1)
- **`--delay`**: Delay between cycles in seconds (default: 5.0)

### How It Works

1. **Multi-Environment Testing**: Miners are tested across multiple environments (SAT, ABD, COIN)
2. **GRPO Scoring**: Group Relative Performance Optimization - scores are relative to group averages
3. **Sybil-Proof Design**: Per-model aggregation collapses Sybil clones into one score
4. **Two-Step Winner Selection**:
   - **Step 1**: Find the best performing model (using aggregate performance from all miners running each model)
   - **Step 2**: Among miners using the winning model, select the one with oldest block commitment (first-come priority)
5. **Winner-Take-All**: Final score = min(environment_scores) - punishes environment imbalance
6. **Smooth Updates**: Exponential moving average reduces score volatility

### Debugging Tools
```bash
# Test miner connectivity and basic info
uv run af test-miners --uids "1,2,3,4,5"
```

## Environments

- **COIN**: Simple coin flip guessing game (easy, ~50% random success)
- **SAT**: Boolean satisfiability problems (medium difficulty logic)
- **ABD**: Program input deduction challenges (hardest, requires reasoning)

Each environment supports automatic sample management with background stock replenishment.