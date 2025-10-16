# Affine Subnet Rewrite

This repository hosts the Affine subnet rewrite: deterministic Gym evaluation environments, duel-only validation, signed sample blocks, and winner-takes-all weight setting. The goal is parity with the legacy subnet while keeping the codebase small and maintainable.

## Quick start

```bash
pip install -e .
pytest
affine --help
```

## Running a validator with Docker Compose

1. Copy `.env.example` to `.env` and fill in your real credentials:
   - `CHUTES_API_KEY`, `AFFINE_CHUTES_URL`
   - `AFFINE_CONTENDER_UID`, `AFFINE_CHAMPION_UID`
   - `AFFINE_SIGNING_KEY_HEX` (ed25519 private key in hex)
   - `AFFINE_BUCKET_*` if you are uploading blocks to R2 / S3 storage
   - `BT_WALLET_*` so weight setting can access your bittensor wallet
2. Ensure your bittensor wallet directory is available at `~/.bittensor/wallets` on the host (the compose file mounts it read-only).
3. Start the stack:

```bash
docker compose up -d --build
```

The `validator` service runs `affine validate` in a loop, uploads blocks if credentials are present, and tails state in `/app/data/blocks` (persisted to the `validator-cache` volume). Watchtower is included so you can point it at a remote tag if you publish the image elsewhere.

### Dry-running weight submission

After blocks exist locally or in the configured bucket, you can compute weights:

```bash
docker compose run --rm validator set-weights --from-bucket --dry-run
```

Drop the `--dry-run` flag once you are comfortable with the computed scores.

## Project layout

- `affine/core.py` – canonical data types and hashing helpers
- `affine/envs.py` – deterministic Gym environments (`mult8-v0`, `tictactoe-v0`)
- `affine/duel.py` – Wilson interval duel logic and ratio schedule
- `affine/validators.py` – validator sampler, block building, VTrust, weights
- `affine/network.py` – thin HTTP client for Chutes
- `affine/storage.py` – R2 / S3 compatible block upload + download
- `affine/cli.py` – Typer CLI (`affine`) providing `validate`, `duel`, `set-weights`

See the `plan/` directory for the original design notes and stretch goals.
