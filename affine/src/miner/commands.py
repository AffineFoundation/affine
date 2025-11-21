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

from affine.core.setup import logger, NETUID
from affine.core.miners import get_miner_info


def get_conf(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get configuration value from environment variable."""
    return os.getenv(key, default)


async def get_miners(uids: Optional[int] = None) -> dict:
    """Get miner information from metagraph.
    
    Args:
        uids: Optional UID to filter (single miner)
    
    Returns:
        Dictionary mapping UID to miner info
    """
    try:
        import bittensor as bt
        
        subtensor = bt.subtensor()
        metagraph = subtensor.metagraph(NETUID)
        
        miners = {}
        
        if uids is not None:
            # Single miner
            if uids < len(metagraph.hotkeys):
                hotkey = metagraph.hotkeys[uids]
                miner_info = await get_miner_info(hotkey, subtensor, NETUID)
                if miner_info:
                    miners[uids] = miner_info
        else:
            # All miners
            for uid in range(len(metagraph.hotkeys)):
                hotkey = metagraph.hotkeys[uid]
                miner_info = await get_miner_info(hotkey, subtensor, NETUID)
                if miner_info:
                    miners[uid] = miner_info
        
        return miners
    
    except Exception as e:
        logger.error(f"Failed to get miners: {e}")
        return {}


async def get_subtensor():
    """Get Bittensor subtensor instance."""
    try:
        import bittensor as bt
        return bt.subtensor()
    except Exception as e:
        logger.error(f"Failed to get subtensor: {e}")
        raise


async def get_latest_chute_id(repo: str, api_key: str) -> Optional[str]:
    """Get latest chute ID for a repository.
    
    Args:
        repo: HF repository name
        api_key: Chutes API key
    
    Returns:
        Chute ID or None if not found
    """
    try:
        import aiohttp
        
        # This is a placeholder - actual implementation would query Chutes API
        # For now, return None to indicate not implemented
        logger.warning("get_latest_chute_id not fully implemented")
        return None
    
    except Exception as e:
        logger.error(f"Failed to get chute ID: {e}")
        return None


async def get_chute(chute_id: str) -> Optional[dict]:
    """Get chute information.
    
    Args:
        chute_id: Chute deployment ID
    
    Returns:
        Chute info dict or None if not found
    """
    try:
        # Placeholder implementation
        logger.warning("get_chute not fully implemented")
        return {"chute_id": chute_id, "status": "unknown"}
    
    except Exception as e:
        logger.error(f"Failed to get chute: {e}")
        return None


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
    
    hf_token = hf_token or get_conf("HF_TOKEN")
    
    miner_map = await get_miners(uids=uid)
    miner = miner_map.get(uid)
    
    if miner is None:
        logger.error(f"No miner found for UID {uid}")
        print(json.dumps({"success": False, "error": f"No miner found for UID {uid}"}))
        sys.exit(1)
    
    repo_name = miner.get("model")
    revision = miner.get("revision")
    
    if not repo_name:
        logger.error(f"Miner {uid} has no model configured")
        print(json.dumps({"success": False, "error": "No model configured"}))
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
        chute_id = await get_latest_chute_id(repo, api_key=chutes_api_key)
        chute_info = await get_chute(chute_id) if chute_id else None
        
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
    wallet = bt.wallet(name=cold, hotkey=hot)
    
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