# Affine

Mine open reasoning.

[Affine Discord](https://discord.com/invite/3T9X4Yn23e) | [FAQ](FAQ.md)

## Introduction

Affine is an incentivized RL environment which pays miners which make incremental improvements on a set of tasks (for instance, program abduction or coding). The mechanism is sybil-proof (you can't cheat by deploying multiple miners), decoy-proof (you can't cheat by packing models into certain environments), copy-proof (you can't cheat by stealing models), overfitting-proof (you can't cheat by overfitting to a single env).

How does Affine work? Affine validators incentivize miners to submit models to Subnet 64 on Bittensor (a.k.a Chutes) where they are inference load balanced and publicly available. These models are evaluated on a set of RL-environments with validators looking for the model which dominates the pareto frontier -- namely the model which outcompetes all other models on all envs (see `af validator`) The network is winners-take-all where miners are forced to copy, download and improve the pareto frontier model.

Why affine? Directed incentives for RL have never been achieved. The ability to direct intelligence and aggregate the work-effort of a large non-permissioned group of individuals on RL tasks will unlock fast advancement in intelligence, we intend to commoditize reasoning (intelligence's highest form) and break the intelligence sound barrier.

## Installation
```bash
# Install uv Astral
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install Affine
git clone https://github.com/AffineFoundation/affine.git
cd affine
uv venv && source .venv/bin/activate && uv pip install -e .

# Verify installation
af
```

### Architecture

Affine now uses [Affinetes](https://github.com/AffineFoundation/affinetes) for container orchestration, providing:
- Clean, lightweight container management
- Support for local and remote Docker deployments
- Environment caching for improved performance
- Type-safe environment definitions

All evaluation environments are packaged as pre-built Docker images, eliminating the need for complex sandbox management.

## Validating
Set env vars, chutes api key, R2 write keys

```bash
# Copy .env and fill out validator items
cp .env.example .env
```

Required environment variables:
```bash
CHUTES_API_KEY=                    # Required for model inference
R2_WRITE_ACCESS_KEY_ID=            # Required for result storage
R2_WRITE_SECRET_ACCESS_KEY=        # Required for result storage
```

(Recommended): Run the validator with docker and watchtower autoupdate.
```bash
# Run the validator with watchtower.
docker-compose down && docker-compose pull && docker-compose up -d && docker-compose logs -f
```
Recreate docker in case of OOM
```bash
docker compose up -d --force-recreate
```
Run the validator using the local override (build local image) + base compose
```bash
docker compose -f docker-compose.yml -f docker-compose.local.yml down --remove-orphans
docker compose -f docker-compose.yml -f docker-compose.local.yml up -d --build --remove-orphans
docker compose -f docker-compose.yml -f docker-compose.local.yml logs -f
```

Run the validator locally(without docker)
```bash
# Start the validator with debug.
af -vv validate
```

# Mining


1. Set env vars.
```bash
# Copy .env and fill out validator items
cp .env.example .env
```

2. Miners need a chutes developer account ( `chutes.ai` ), and you must fund your Chutes account to deploy miners.

```bash
chutes register
```

After registering, you will need to fund your Chutes account with $TAO.
Your Chutes payment address can be found in `~/.chutes/config.ini`.
Send TAO to this address before deploying models.


3. Register your miner to Affine (S120).
```bash
btcli subnet register --wallet.name <your cold> --wallet.hotkey <your hot>
```

4. Pull a model off the network.
```bash
af -vvv pull <uid to pull> --model_path <i.e. ./my_model>
```

5. Improve the model
```bash
... magic RL stuff ...
```

6. Upload your model to Hugging Face (manual, required before deploying).
   - Create or choose an existing model repo (e.g. `<user>/Affine-<repo>`)
   - Push your model artifacts and obtain the commit SHA you wish to deploy
   - You are responsible for the HF upload process (e.g. `huggingface-cli`, `git lfs`)

7. Deploy the HF repo+revision to Chutes.
```bash
af -vvv chutes_push --repo <user/repo> --revision <sha> --chutes-api-key ...

### Configure Chutes deployment settings

You can customize how your Chute is deployed (GPU type, concurrency, scaling, etc.) by editing the Chutes config we generate in code.

- Open `affine/affine/cli.py`
- Find the `deploy_to_chutes()` function inside the `chutes_push` command
- Edit the arguments passed to `build_sglang_chute(...)` to match your needs

Refer to the official Chutes documentation for all available options and best practices: [chutesai/chutes](https://github.com/chutesai/chutes).


```
This prints a JSON payload including `chute_id`. Keep it for the next step.

8. Commit the deployment on-chain (separate from deployment).
```bash
af -vvv commit --repo <user/repo> --revision <sha> --chute-id <chute_id> --coldkey <your cold> --hotkey <your hot>
```

## Sampling Scheduler

Affine uses an advanced sampling scheduler that provides efficient, independent sampling for each miner with dynamic rate adjustment and comprehensive monitoring.

**Key Features:**
- Independent sampling coroutines per miner (95%+ concurrency utilization)
- Dynamic rate adjustment (3x acceleration for miners with <200 samples)
- Backpressure control with queue management
- Intelligent Chutes error detection and auto-pause
- Real-time monitoring API

**Documentation:**
- [Sampling Scheduler Guide](docs/SAMPLING_SCHEDULER.md) - Architecture, configuration, and usage
- [Monitoring API Reference](docs/MONITORING_API.md) - HTTP endpoints for status monitoring
- [Scheduler Initialization Design](docs/SCHEDULER_INITIALIZATION_DESIGN.md) - Historical data loading design

**Quick Start:**
```bash
# Start the sampling scheduler
af runner

# Enable with monitoring API (recommended)
af runner --enable-monitoring
```

# SDK
Affine is also an SDK you can use to evaluate models across different environments.

```python
import asyncio
import affine as af
from dotenv import load_dotenv

# Optionally turn on logging
af.trace()  # or af.debug() or af.info()

load_dotenv(override=True)

async def main():
    # Get miner info for a specific UID
    # NOTE: CHUTES_API_KEY environment variable is required
    miner_dict = await af.miners(160)
    miner = miner_dict.get(160)
    assert miner, "Unable to obtain miner, please check if registered"

    # Evaluate on Affine environments
    ded_env = af.DED()
    evaluation = await ded_env.evaluate(miner)
    print("Score:", evaluation.score)
    print("Extra:", evaluation.extra)

    # Evaluate on AgentGym environments with task IDs
    alfworld_env = af.ALFWORLD()
    
    # Random task (default)
    evaluation = await alfworld_env.evaluate(miner)
    
    # Specific single task
    evaluation = await alfworld_env.evaluate(miner, task_id=10)
    
    # Multiple tasks
    evaluation = await alfworld_env.evaluate(miner, task_id=[0, 1, 2])
    
    # You can also pass parameters directly without a miner object
    evaluation = await ded_env.evaluate(
        model="deepseek-ai/DeepSeek-V3",
        base_url="https://llm.chutes.ai/v1",
        temperature=0.7
    )

    # List all available environments
    envs = af.tasks.list_available_environments()
    for env_type, env_names in envs.items():
        print(f"\n{env_type}:")
        for name in env_names:
            print(f"  - {name}")

    # Async generator of results from last 100 blocks
    async for res in af.dataset(100):
        print(res)

if __name__ == "__main__":
    asyncio.run(main())
```
