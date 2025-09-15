#!/usr/bin/env python3
# --------------------------------------------------------------------------- #
#                             Imports                                         #
# --------------------------------------------------------------------------- #
from __future__ import annotations
import os
import sys
import json
import click
import asyncio
import textwrap
from .utils import *
import bittensor as bt
from pathlib import Path
from huggingface_hub import HfApi
from huggingface_hub import snapshot_download
from bittensor.core.errors import MetadataError
from typing import Any, Dict, List, Optional, Union, Tuple, Sequence, Literal, TypeVar, Awaitable

import affine as af
# --------------------------------------------------------------------------- #
#                              Pull Model                                     #
# --------------------------------------------------------------------------- #
@af.cli.command("pull")
@click.argument("uid", type=int)
@click.option("--model_path", "-p", default = './model_path', required=True, type=click.Path(), help="Local directory to save the model")
@click.option('--hf-token', default=None, help="Hugging Face API token (env HF_TOKEN if unset)")
def pull(uid: int, model_path: str, hf_token: str):
    """Pulls a model from a specific miner UID if exists."""

    # 1. Ensure HF token
    hf_token     = hf_token or af.get_conf("HF_TOKEN")

    # 2. Lookup miner on‑chain
    miner_map = asyncio.run(af.get_miners(uids=uid))
    miner = miner_map.get(uid)
    
    if miner is None:
        click.echo(f"No miner found for UID {uid}", err=True)
        sys.exit(1)
    repo_name = miner.model
    af.logger.info("Pulling model %s for UID %d into %s", repo_name, uid, model_path)

    # 3. Download snapshot
    try:
        snapshot_download(
            repo_id=repo_name,
            repo_type="model",
            local_dir=model_path,
            token=hf_token,
            resume_download=True,
            revision=miner.revision,
        )
        click.echo(f"Model {repo_name} pulled to {model_path}")
    except Exception as e:
        af.logger.error("Failed to download %s: %s", repo_name, e)
        click.echo(f"Error pulling model: {e}", err=True)
        sys.exit(1)


# --------------------------------------------------------------------------- #
#                              Push Model                                     #
# --------------------------------------------------------------------------- #
@af.cli.command("push")
@click.option('--model_path',  default='./model_path', help='Local path to model artifacts.')
@click.option('--existing-repo', default=None, help='Use an existing HF repo instead of uploading (format <user>/<repo>)')
@click.option('--revision', default=None, help='Commit SHA to register (only relevant with --existing-repo)')
@click.option('--coldkey',     default=None, help='Name of the cold wallet to use.')
@click.option('--hotkey',      default=None, help='Name of the hot wallet to use.')
@click.option('--chutes-api-key', default=None, help='Chutes API key (env CHUTES_API_KEY if unset)')
def push(model_path: str, existing_repo: str, revision: str, coldkey: str, hotkey: str, chutes_api_key: str):
    """Pushes a model to be hosted by your miner."""
    # -----------------------------------------------------------------------------
    # 1. Wallet & config
    # -----------------------------------------------------------------------------
    coldkey = coldkey or af.get_conf("BT_WALLET_COLD", "default")
    hotkey  = hotkey  or af.get_conf("BT_WALLET_HOT", "default")
    af.logger.debug("Using coldkey=%s, hotkey=%s", coldkey, hotkey)
    wallet = bt.wallet(name=coldkey, hotkey=hotkey)

    # Required API credentials
    hf_user        = af.get_conf("HF_USER")
    hf_token       = af.get_conf("HF_TOKEN")
    chutes_api_key = chutes_api_key or af.get_conf("CHUTES_API_KEY")
    chute_user     = af.get_conf("CHUTE_USER")
    # TODO: validate API creds, exit gracefully if missing

    # -----------------------------------------------------------------------------
    # 2. Prepare HF repo name - If --existing-repo provided, use it directly and skip local upload
    # -----------------------------------------------------------------------------
    repo_name = existing_repo or f"{hf_user}/Affine-{wallet.hotkey.ss58_address}"
    af.logger.debug("Using existing HF repo: %s" if existing_repo else "Hugging Face repo: %s", repo_name)

    # -----------------------------------------------------------------------------
    # 3. Create & secure HF repo
    # -----------------------------------------------------------------------------
    api = HfApi(token=hf_token)
    if not existing_repo:
        api.create_repo(repo_id=repo_name, repo_type="model", private=True, exist_ok=True)
        try: api.update_repo_visibility(repo_id=repo_name, private=True)
        except Exception: af.logger.debug("Repo already private or visibility update failed")

    # -----------------------------------------------------------------------------
    # 4. Upload model files to HF (skip if using existing repo)
    # -----------------------------------------------------------------------------
    async def deploy_model_to_hf():
        af.logger.debug("Starting model upload from %s", model_path)
        # Gather files
        files = []
        for root, _, fnames in os.walk(model_path):
            if ".cache" in root or any(p.startswith(".") for p in root.split(os.sep)):
                continue
            for fname in fnames:
                if not (fname.startswith(".") or fname.endswith(".lock")):
                    files.append(os.path.join(root, fname))

        # Upload files with limited concurrency to avoid HF 429 errors
        SEM = asyncio.Semaphore(int(os.getenv("AFFINE_UPLOAD_CONCURRENCY", "2")))

        async def _upload(path: str):
            rel = os.path.relpath(path, model_path)
            async with SEM:  # limit concurrent commits
                await asyncio.to_thread(
                    lambda: api.upload_file(
                        path_or_fileobj=path,
                        path_in_repo=rel,
                        repo_id=repo_name,
                        repo_type="model"
                    )
                )
                af.logger.debug("Uploaded %s", rel)

        await asyncio.gather(*(_upload(p) for p in files))
        af.logger.debug("Model upload complete (%d files)", len(files))

    asyncio.run(deploy_model_to_hf()) if not existing_repo else af.logger.debug("Skipping model upload because --existing-repo was provided")

    # -----------------------------------------------------------------------------
    # 5. Fetch latest revision hash
    # -----------------------------------------------------------------------------
    if revision:
        af.logger.debug("Using user-supplied revision: %s", revision)
    else:
        info      = api.repo_info(repo_id=repo_name, repo_type="model")
        revision  = getattr(info, "sha", getattr(info, "oid", "")) or ""
        af.logger.debug("Latest revision from HF: %s", revision)

    # -----------------------------------------------------------------------------
    # 6. Commit model revision on-chain
    # -----------------------------------------------------------------------------
    chute_id = None

    async def commit_to_chain():
        """Submit the model commitment, retrying on quota errors."""
        af.logger.debug("Preparing on-chain commitment")
        sub     = await af.get_subtensor()
        payload = json.dumps({"model": repo_name, "revision": revision, "chute_id": chute_id})
        while True:
            try:
                await sub.set_reveal_commitment(wallet=wallet, netuid=af.NETUID, data=payload, blocks_until_reveal=1)
                af.logger.debug("On-chain commitment submitted")
                break
            except MetadataError as e:
                if "SpaceLimitExceeded" in str(e):
                    af.logger.debug("SpaceLimitExceeded – waiting one block before retrying")
                    await sub.wait_for_block()
                else:
                    raise


    # -----------------------------------------------------------------------------
    # 7. Make HF repo public
    # -----------------------------------------------------------------------------
    try:
        api.update_repo_visibility(repo_id=repo_name, private=False)
        af.logger.debug("Repo made public")
    except Exception:
        af.logger.trace("Failed to make repo public (already public?)")

    # -----------------------------------------------------------------------------
    # 8. Deploy Chute
    # -----------------------------------------------------------------------------
    async def deploy_to_chutes():
        af.logger.debug("Building Chute config")
        rev_flag = f'revision="{revision}",' if revision else ""
        chutes_config = textwrap.dedent(f"""
import os
from chutes.chute import NodeSelector
from chutes.chute.template.sglang import build_sglang_chute
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

chute = build_sglang_chute(
    username="{chute_user}",
    readme="{repo_name}",
    model_name="{repo_name}",
    image="chutes/sglang:0.4.9.post3",
    concurrency=20,
    {rev_flag}
    node_selector=NodeSelector(
        gpu_count=8,
        min_vram_gb_per_gpu=24,
    ),
    engine_args=(
        "--trust-remote-code "
    ),
)
""")
        tmp_file = Path("tmp_chute.py")
        tmp_file.write_text(chutes_config)
        af.logger.debug("Wrote Chute config to %s", tmp_file)
        af.logger.debug("=== chute file ===\n%s", tmp_file.read_text())

        cmd = ["chutes", "deploy", f"{tmp_file.stem}:chute", "--public"]
        env = {**os.environ, "CHUTES_API_KEY": chutes_api_key}
        proc = await asyncio.create_subprocess_exec(
            *cmd, env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.PIPE,
        )
        # Auto-answer the interactive Y/N prompt
        if proc.stdin:
            proc.stdin.write(b"y\n")
            await proc.stdin.drain()
            proc.stdin.close()
        stdout, _ = await proc.communicate()
        output = stdout.decode().split('confirm? (y/n)')[1].strip()
        af.logger.trace(output)

        import re
        match = re.search(r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3})\s+\|\s+(\w+)', output)
        if match and match.group(2) == "ERROR":
            af.logger.debug("Chutes deploy failed with the above error log")
            raise RuntimeError("Chutes deploy failed")
        if proc.returncode != 0:
            af.logger.debug("Chutes deploy failed with code %d", proc.returncode)
            raise RuntimeError("Chutes deploy failed")
        tmp_file.unlink(missing_ok=True)
        af.logger.debug("Chute deployment successful")

    asyncio.run(deploy_to_chutes())

    # -----------------------------------------------------------------------------
    # 8b. Retrieve chute_id and commit on-chain
    # -----------------------------------------------------------------------------
    chute_id = asyncio.run(af.get_latest_chute_id(repo_name, api_key=chutes_api_key))

    asyncio.run(commit_to_chain())

    # -----------------------------------------------------------------------------
    # 9. Warm up model until it’s marked hot
    # -----------------------------------------------------------------------------
    async def warmup_model():
        af.logger.debug("Warming up model with SAT challenges")
        sub       = await af.get_subtensor()
        meta      = await sub.metagraph(af.NETUID)
        my_uid    = meta.hotkeys.index(wallet.hotkey.ss58_address)
        miner  = (await af.get_miners(netuid=af.NETUID))[my_uid]

        while not (miner.chute or {}).get("hot", False):
            challenge = await af.SAT().generate()
            await af.run(challenges=challenge, miners=[miner])
            await sub.wait_for_block()
            miner = (await af.get_miners(netuid=af.NETUID))[my_uid]
            af.logger.trace("Checked hot status: %s", (miner.chute or {}).get("hot"))

        af.logger.debug("Model is now hot and ready")

    asyncio.run(warmup_model())
    af.logger.debug("Mining setup complete. Model is live!")  
