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

## Environments

- **COIN**: Simple coin flip guessing game
- **SAT**: Boolean satisfiability problems
- **ABD**: Program input deduction challenges

Each environment supports automatic sample management with background stock replenishment.