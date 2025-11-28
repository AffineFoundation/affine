# Affine Miner Guide

This document provides a complete guide for mining on the Affine subnet (Subnet 120).

## Table of Contents

- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Mining Workflow](#mining-workflow)
- [CLI Reference](#cli-reference)
- [Common Issues](#common-issues)

## Prerequisites

1. **Hugging Face Account**: For hosting models
2. **Chutes.ai Account**: Register using the **same hotkey** as your mining hotkey (no developer deposit required)
3. **Chutes Account Funding**: TAO required to pay for GPU time when your model is running
4. **Bittensor Wallet**: Hotkey registered to Subnet 120

## Environment Setup

### 1. Install Affine

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install Affine
git clone https://github.com/AffineFoundation/affine.git
cd affine
uv venv && source .venv/bin/activate && uv pip install -e .

# Verify installation
af
```

### 2. Configure Environment Variables

Copy and edit the `.env` file:

```bash
cp .env.example .env
```

Edit `.env` file with required variables:

```bash
# Chutes API key (get from chutes.ai, format: cpk_...)
CHUTES_API_KEY=cpk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx.xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx.xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Bittensor wallet configuration
BT_WALLET_COLD=your_coldkey_name
BT_WALLET_HOT=your_hotkey_name

# Hugging Face token (needs Write permission to upload models)
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Chutes username
CHUTE_USER=your_username

# Subtensor configuration (optional)
SUBTENSOR_ENDPOINT=finney
SUBTENSOR_FALLBACK=wss://lite.sub.latent.to:443
```

### 3. Register Chutes Account

```bash
chutes register
```

After registration, view your payment address and fund it with TAO:

```bash
cat ~/.chutes/config.ini
```

Send TAO to the displayed address to pay for Chute running costs.

### 4. Register to Subnet 120

```bash
btcli subnet register --wallet.name <your_coldkey> --wallet.hotkey <your_hotkey>
```

## Mining Workflow

### Step 1: Pull an Existing Model

Pull an existing model from the network as a starting point:

```bash
af pull <UID> --model-path ./my_model
```

**Parameters:**
- `<UID>`: UID of the miner to pull from
- `--model-path`: Local directory path to save the model (optional, default: `./model_path`)
- `--hf-token`: Hugging Face token (optional, reads from environment variable)

**Example:**
```bash
# Pull model from UID 42 to ./my_model directory
af pull 42 --model-path ./my_model
```

### Step 2: Improve the Model

Improve model performance using reinforcement learning or other methods. This is the core of mining:

- Train the model on Affine's evaluation environments
- Optimize model performance across multiple tasks
- Ensure your model is competitive on the Pareto frontier

### Step 3: Upload Model to Hugging Face

Manually upload your improved model to Hugging Face:

1. Create or select an HF repository (e.g., `<username>/Affine-<repo>`)
2. Push your model using `huggingface-cli` or `git lfs`
3. Obtain the commit SHA

**Example:**
```bash
# Upload using huggingface-cli
huggingface-cli upload <username>/Affine-model ./my_model

# Or using git
cd ./my_model
git init
git lfs install
git lfs track "*.safetensors"
git add .
git commit -m "Initial model commit"
git remote add origin https://huggingface.co/<username>/Affine-model
git push origin main
```

### Step 4: Deploy to Chutes

Deploy your Hugging Face model as a Chute:

```bash
af chutes_push --repo <username/repo> --revision <SHA>
```

**Parameters:**
- `--repo`: Hugging Face repository ID (required)
- `--revision`: Git commit SHA (required)
- `--chutes-api-key`: Chutes API key (optional, reads from environment variable)
- `--chute-user`: Chutes username (optional, reads from environment variable)

**Example:**
```bash
af chutes_push --repo myuser/Affine-model --revision abc123def456
```

This command outputs a JSON response containing `chute_id`. Save this ID for the next step.

**Customize Chute Configuration:**

To customize deployment settings (GPU type, concurrency, etc.), edit the `deploy_to_chutes()` function in [`affine/affine/cli.py`](../affine/cli.py:124).

Refer to the [official Chutes documentation](https://github.com/chutesai/chutes) for all configuration options.

### Step 5: Commit On-Chain

Commit the deployment information to the blockchain:

```bash
af commit --repo <username/repo> --revision <SHA> --chute-id <chute_id>
```

**Parameters:**
- `--repo`: Hugging Face repository ID (required)
- `--revision`: Git commit SHA (required)
- `--chute-id`: Chutes deployment ID (required)
- `--coldkey`: Coldkey name (optional, reads from environment variable)
- `--hotkey`: Hotkey name (optional, reads from environment variable)

**Example:**
```bash
af commit --repo myuser/Affine-model --revision abc123def456 --chute-id chute_789xyz
```

## CLI Reference

### Query Commands

#### Query Sample Result

```bash
af get-sample <UID> <environment> <task_id>
```

Query sample result for a specific miner on a specific environment and task.

**Examples:**
```bash
af get-sample 42 affine:ded task_123
af get-sample 100 agentgym:webshop 456
```

#### Query Miner Information

```bash
af get-miner <UID>
```

Query complete miner information including hotkey, model, revision, chute_id, validation status, and timestamps.

**Example:**
```bash
af get-miner 42
```

#### Query Weights

```bash
af get-weights
```

Query the latest normalized weights for on-chain weight setting.

#### Query Scores

```bash
af get-scores [--top N]
```

Query top N miners by score.

**Parameters:**
- `--top, -t`: Return top N miners (default: 10)

**Examples:**
```bash
af get-scores
af get-scores --top 20
```

#### Query Task Pool

```bash
af get-pool <UID> <environment> [--full]
```

Query the list of pending task IDs for a miner in a specific environment.

**Parameters:**
- `--full`: Display full task ID list without truncation

**Examples:**
```bash
af get-pool 100 agentgym:webshop
af get-pool 100 agentgym:webshop --full
```

#### Query Ranking Table

```bash
af get-rank
```

Fetch and display the latest miner ranking table in the same format as scorer output.

### Verbose Logging

All commands support increased logging verbosity:

```bash
# INFO level
af -v pull 42

# DEBUG level
af -vv commit --repo myuser/model --revision abc123 --chute-id xyz789

# TRACE level
af -vvv chutes_push --repo myuser/model --revision abc123
```

## Common Issues

### Q: My Chute is "cold" and not receiving validator requests?

**A:** Chutes automatically shut down after a period of inactivity (default 5-10 minutes) to save costs. Solutions:

1. **Increase shutdown time**: Increase the `shutdown_after_seconds` parameter in your Chute config (e.g., set to `1800` for 30 minutes)
2. **Keep it warm**: Write a script to send requests to your Chute every few minutes to keep it active

### Q: Chute deployment fails or won't activate?

**A:** This is usually a configuration issue:

1. **Check logs**: Retrieve live logs using Instance ID and Chutes API key (detailed guide on Discord)
2. **Common errors**:
   - Invalid `engine_args` parameters or comma errors
   - Outdated image version, need to update `image` parameter
   - Corrupted `model.safetensors` file or upload error

### Q: New model doesn't appear on leaderboard?

**A:** The system needs time to evaluate models (e.g., 10,000 blocks, which can be over a day). If still not appearing:

1. Confirm on-chain commit was successful
2. Check that the committed model revision matches the Chute deployment

### Q: How to view my model's performance across environments?

**A:** Use query commands:

```bash
# View rankings
af get-rank

# View sample results for specific environment
af get-sample <your_UID> agentgym:webshop <task_id>

# View pending tasks
af get-pool <your_UID> agentgym:webshop
```

### Q: Can environment variables be overridden via command-line arguments?

**A:** Yes, most commands support overriding environment variables with parameters:

```bash
# Using command-line arguments
af commit --repo myuser/model --revision abc123 --chute-id xyz789 --coldkey mywallet --hotkey myhotkey

# Or rely on environment variables
af commit --repo myuser/model --revision abc123 --chute-id xyz789
```

## Related Documentation

- [Main Documentation](../README.md) - Affine project overview
- [Validator Guide](VALIDATOR.md) - Validator operation guide
- [FAQ](FAQ.md) - Frequently Asked Questions
- [Chutes Documentation](https://github.com/chutesai/chutes) - Official Chutes platform documentation