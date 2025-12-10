"""
Miner Commands Implementation

Provides command functions for miners:
- commit_command: Commit model to blockchain
- pull_command: Pull model from Hugging Face
- chutes_push_command: Deploy model to Chutes
"""

import os
import sys
import json
import asyncio
import textwrap
from pathlib import Path
from typing import Optional
from affine.utils.api_client import create_api_client, cli_api_client
from affine.core.setup import logger, NETUID


def get_conf(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get configuration value from environment variable."""
    return os.getenv(key, default)


async def get_subtensor():
    """Get Bittensor subtensor instance."""
    try:
        import bittensor as bt
        return bt.subtensor()
    except Exception as e:
        logger.error(f"Failed to get subtensor: {e}")
        raise


# ============================================================================
# Command Implementations
# ============================================================================

async def pull_command(uid: int, model_path: str, hf_token: Optional[str] = None):
    """Pull model from Hugging Face.
    
    Args:
        uid: Miner UID
        model_path: Local directory to save model
        hf_token: Hugging Face API token (optional, from env if not provided)
    """
    from huggingface_hub import snapshot_download
    from affine.utils.subtensor import get_subtensor

    hf_token = hf_token or get_conf("HF_TOKEN")
    
    # Get miner info directly from subtensor
    try:
        subtensor = await get_subtensor()
        meta = await subtensor.metagraph(NETUID)
        commits = await subtensor.get_all_revealed_commitments(NETUID)
        
        if uid >= len(meta.hotkeys):
            logger.error(f"Invalid UID {uid}")
            print(json.dumps({"success": False, "error": f"Invalid UID {uid}"}))
            sys.exit(1)
        
        hotkey = meta.hotkeys[uid]
        
        if hotkey not in commits:
            logger.error(f"No commit found for UID {uid}")
            print(json.dumps({"success": False, "error": f"No commit found for UID {uid}"}))
            sys.exit(1)
        
        _, commit_data = commits[hotkey][-1]
        data = json.loads(commit_data)
        
        repo_name = data.get("model")
        revision = data.get("revision")
        
        if not repo_name:
            logger.error(f"Miner {uid} has no model configured")
            print(json.dumps({"success": False, "error": "No model configured"}))
            sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to get miner info: {e}")
        print(json.dumps({"success": False, "error": str(e)}))
        sys.exit(1)
    
    logger.info(f"Pulling model {repo_name}@{revision} for UID {uid} into {model_path}")
    
    try:
        snapshot_download(
            repo_id=repo_name,
            repo_type="model",
            local_dir=model_path,
            token=hf_token,
            resume_download=True,
            revision=revision,
        )
        
        result = {
            "success": True,
            "uid": uid,
            "repo": repo_name,
            "revision": revision,
            "path": model_path
        }
        print(json.dumps(result))
        logger.info(f"Model {repo_name} pulled successfully")
    
    except Exception as e:
        logger.error(f"Failed to download {repo_name}: {e}")
        print(json.dumps({"success": False, "error": str(e)}))
        sys.exit(1)


async def get_latest_chute_id(repo: str, api_key: str) -> Optional[str]:
    """Get latest chute ID for a repository.
    
    Args:
        repo: HF repository name
        api_key: Chutes API key
    
    Returns:
        Chute ID or None if not found
    """
    token = api_key or os.getenv("CHUTES_API_KEY", "")
    if not token:
        return None
    
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://api.chutes.ai/chutes/", headers={"Authorization": token}
            ) as r:
                if r.status != 200:
                    return None
                data = await r.json()
    except Exception:
        return None
    
    chutes = data.get("items", data) if isinstance(data, dict) else data
    if not isinstance(chutes, list):
        return None
    
    for chute in reversed(chutes):
        if any(chute.get(k) == repo for k in ("model_name", "name", "readme")):
            return chute.get("chute_id")
    return None


async def chutes_push_command(
    repo: str,
    revision: str,
    chutes_api_key: Optional[str] = None,
    chute_user: Optional[str] = None
):
    """Deploy model to Chutes.
    
    Args:
        repo: HF repository ID
        revision: HF commit SHA
        chutes_api_key: Chutes API key (optional, from env if not provided)
        chute_user: Chutes username (optional, from env if not provided)
    """
    chutes_api_key = chutes_api_key or get_conf("CHUTES_API_KEY")
    chute_user = chute_user or get_conf("CHUTE_USER")
    
    if not chutes_api_key:
        logger.error("CHUTES_API_KEY not configured")
        print(json.dumps({"success": False, "error": "CHUTES_API_KEY not configured"}))
        sys.exit(1)
    
    if not chute_user:
        logger.error("CHUTE_USER not configured")
        print(json.dumps({"success": False, "error": "CHUTE_USER not configured"}))
        sys.exit(1)
    
    logger.debug(f"Building Chute config for repo={repo} revision={revision}")
    
    # Generate Chute configuration
    chutes_config = textwrap.dedent(
        f"""
import os
from chutes.chute import NodeSelector
from chutes.chute.template.sglang import build_sglang_chute
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

chute = build_sglang_chute(
    username="{chute_user}",
    readme="{repo}",
    model_name="{repo}",
    image="chutes/sglang:nightly-2025081600",
    concurrency=20,
    revision="{revision}",
    node_selector=NodeSelector(
        gpu_count=1,
        include=["a100", "h100"],
    ),
    max_instances=1,
    scale_threshold=0.5,
    shutdown_after_seconds=3600,
)
"""
    )
    
    tmp_file = Path("tmp_chute.py")
    tmp_file.write_text(chutes_config)
    logger.debug(f"Wrote Chute config to {tmp_file}")
    
    # Deploy to Chutes
    cmd = ["chutes", "deploy", f"{tmp_file.stem}:chute", "--accept-fee"]
    env = {**os.environ, "CHUTES_API_KEY": chutes_api_key}
    
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.PIPE,
        )
        
        if proc.stdin:
            proc.stdin.write(b"y\n")
            await proc.stdin.drain()
            proc.stdin.close()
        
        stdout, _ = await proc.communicate()
        output = stdout.decode(errors="ignore")
        logger.trace(output)
        
        # Check for errors
        import re
        match = re.search(
            r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3})\s+\|\s+(\w+)", output
        )
        if match and match.group(2) == "ERROR":
            logger.debug("Chutes deploy failed with error log")
            raise RuntimeError("Chutes deploy failed")
        
        if proc.returncode != 0:
            logger.debug(f"Chutes deploy failed with code {proc.returncode}")
            raise RuntimeError("Chutes deploy failed")
        
        tmp_file.unlink(missing_ok=True)
        logger.debug("Chute deployment successful")
        
        # Get chute info
        from affine.utils.api_client import get_chute_info
        chute_id = await get_latest_chute_id(repo, api_key=chutes_api_key)
        chute_info = await get_chute_info(chute_id) if chute_id else None
        
        result = {
            "success": bool(chute_id),
            "chute_id": chute_id,
            "chute": chute_info,
            "repo": repo,
            "revision": revision,
        }
        print(json.dumps(result))
        logger.info(f"Deployed to Chutes: {chute_id}")
    
    except Exception as e:
        logger.error(f"Chutes deployment failed: {e}")
        print(json.dumps({"success": False, "error": str(e)}))
        tmp_file.unlink(missing_ok=True)
        sys.exit(1)


async def commit_command(
    repo: str,
    revision: str,
    chute_id: str,
    coldkey: Optional[str] = None,
    hotkey: Optional[str] = None
):
    """Commit model to blockchain.
    
    Args:
        repo: HF repository ID
        revision: HF commit SHA
        chute_id: Chutes deployment ID
        coldkey: Wallet coldkey name (optional, from env if not provided)
        hotkey: Wallet hotkey name (optional, from env if not provided)
    """
    import bittensor as bt
    from bittensor.core.errors import MetadataError
    
    cold = coldkey or get_conf("BT_WALLET_COLD", "default")
    hot = hotkey or get_conf("BT_WALLET_HOT", "default")
    wallet = bt.Wallet(name=cold, hotkey=hot)
    
    logger.info(f"Committing: {repo}@{revision} (chute: {chute_id})")
    logger.info(f"Using wallet: {wallet.hotkey.ss58_address[:16]}...")
    
    async def _commit():
        sub = await get_subtensor()
        data = json.dumps({
            "model": repo,
            "revision": revision,
            "chute_id": chute_id
        })
        
        while True:
            try:
                await sub.set_reveal_commitment(
                    wallet=wallet,
                    netuid=NETUID,
                    data=data,
                    blocks_until_reveal=1
                )
                break
            except MetadataError as e:
                if "SpaceLimitExceeded" in str(e):
                    logger.warning("Space limit exceeded, waiting for next block...")
                    await sub.wait_for_block()
                else:
                    raise
    
    try:
        await _commit()
        
        result = {
            "success": True,
            "repo": repo,
            "revision": revision,
            "chute_id": chute_id,
        }
        print(json.dumps(result))
        logger.info("Commit successful")
    
    except Exception as e:
        logger.error(f"Commit failed: {e}")
        print(json.dumps({"success": False, "error": str(e)}))
        sys.exit(1)


async def get_sample_command(
    uid: int,
    env: str,
    task_id: str,
):
    """Query sample result by UID, environment, and task ID.
    
    Args:
        uid: Miner UID
        env: Environment name
        task_id: Task ID
    """
    
    async with cli_api_client() as client:
        endpoint = f"/samples/uid/{uid}/{env}/{task_id}"
        data = await client.get(endpoint)
        
        if data:
            print(json.dumps(data, indent=2, ensure_ascii=False))


async def get_miner_command(uid: int):
    """Query miner status and information by UID.
    
    Args:
        uid: Miner UID
    """
    async with cli_api_client() as client:
        endpoint = f"/miners/uid/{uid}"
        data = await client.get(endpoint)
        if data:
            print(json.dumps(data, indent=2, ensure_ascii=False))



async def get_weights_command():
    """Query latest normalized weights for on-chain weight setting.
    
    Returns the most recent score snapshot with normalized weights
    for all miners.
    """
    async with cli_api_client() as client:
        endpoint = "/scores/weights/latest"
        data = await client.get(endpoint)
        
        if data:
            print(json.dumps(data, indent=2, ensure_ascii=False))


async def get_scores_command(top: int = 32):
    """Query latest scores for top N miners.
    
    Args:
        top: Number of top miners to return (default: 256)
    """
    async with cli_api_client() as client:
        endpoint = f"/scores/latest?top={top}"
        data = await client.get(endpoint)
        
        if data:
            print(json.dumps(data, indent=2, ensure_ascii=False))


async def get_score_command(uid: int):
    """Query score for a specific miner by UID.
    
    Args:
        uid: Miner UID
    """
    async with cli_api_client() as client:
        endpoint = f"/scores/uid/{uid}"
        data = await client.get(endpoint)
        
        if data:
            print(json.dumps(data, indent=2, ensure_ascii=False))


async def get_pool_command(uid: int, env: str, full: bool = False):
    """Query task pool status for a miner in an environment.
    
    Args:
        uid: Miner UID
        env: Environment name (e.g., agentgym:webshop)
        full: If True, print full task_ids lists without truncation
    """
    async with cli_api_client() as client:
        endpoint = f"/samples/pool/uid/{uid}/{env}"
        data = await client.get(endpoint)
        
        if data:
            if data.get("success") is False:
                print(json.dumps({
                    "error": data.get("error"),
                    "status_code": data.get("status_code")
                }, indent=2, ensure_ascii=False))
                return
            if full:
                # Print full data without truncation
                print(json.dumps(data, indent=2, ensure_ascii=False))
            else:
                # Format output for better readability
                # Show summary first, then task_ids ranges instead of full lists
                summary = {
                    "uid": data.get("uid"),
                    "hotkey": data.get("hotkey"),
                    "model_revision": data.get("model_revision"),
                    "env": data.get("env"),
                    "sampling_range": data.get("sampling_range"),
                    "total_tasks": data.get("total_tasks"),
                    "sampled_count": data.get("sampled_count"),
                    "pool_count": data.get("pool_count"),
                    "missing_count": data.get("missing_count"),
                }
                
                # Helper function to format task_id list as ranges
                def format_task_ids(task_ids):
                    if not task_ids:
                        return "[]"
                    if len(task_ids) <= 10:
                        return str(task_ids)
                    # Show first 5 and last 5
                    return f"[{', '.join(map(str, task_ids[:5]))}, ..., {', '.join(map(str, task_ids[-5:]))}] (total: {len(task_ids)})"
                
                summary["sampled_task_ids"] = format_task_ids(data.get("sampled_task_ids", []))
                summary["pool_task_ids"] = format_task_ids(data.get("pool_task_ids", []))
                summary["missing_task_ids"] = format_task_ids(data.get("missing_task_ids", []))
                
                print(json.dumps(summary, indent=2, ensure_ascii=False))