# Affine

Mine open reasoning.

[Affine Discord](https://discord.com/invite/3T9X4Yn23e)

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

## Validating
Set env vars, chutes api key, R2 write keys

```bash
# Copy .env and fill out validator items
cp .env.example .env
```

```bash
CHUTES_API_KEY=
.
.
.
R2_WRITE_ACCESS_KEY_ID=
R2_WRITE_SECRET_ACCESS_KEY=
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

IMPORTANT: you require a ***developer enabled account*** on Chutes to mine. Normal API keys cannot deploy chutes right now.

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

# SDK
Affine is also an SDK you can use to generate and evaluate models envs.
```python
import affine as af

# Optionally turn on logging 
af.trace(); af.debug(); af.info()

# Get all miner info or only for UID =5
miners = await af.miners()
miner = await af.miners( 5 )

# Generate a SAT challenge
chal = await af.SAT.generate() 

# Generate a bunch.
chals = await af.ABDUCTION().many( 10 )
chals = await af.DEDUCTION().many( 10 )

# Query the model directly.
# NOTE: A CHUTES_API_KEY .env value is required for this command.
response = await af.query( chal.prompt, model = miner.model )

# Evaluate the response
evaluation = chal.evaluate( response ) 
print( evaluation.score )

# Async generator of results from last 100 blocks.
async for res in af.dataset(100):
    print (res)          # Result objects
```
