"""Blockchain weight setting utilities.

This module contains functions for setting weights on the Bittensor blockchain,
including retry mechanisms, confirmation verification, and signer integration.
"""

import os
import time
import socket
import asyncio
import logging
import aiohttp
import bittensor as bt
from typing import List
from urllib.parse import urlparse
from aiohttp import ClientConnectorError

from affine.config import get_conf
from affine.setup import NETUID, logger
from affine.utils.subtensor import get_subtensor


async def _verify_weights_updated(
    wallet: "bt.wallet",
    netuid: int,
    ref_block: int,
    log_prefix: str,
) -> bool:
    """Verify that weights have been updated on chain.
    
    Args:
        wallet: Bittensor wallet
        netuid: Network UID
        ref_block: Reference block number from submission
        log_prefix: Logging prefix
        
    Returns:
        True if weights were updated, False otherwise
    """
    st = await get_subtensor()
    meta = await st.metagraph(netuid)
    
    try:
        idx = meta.hotkeys.index(wallet.hotkey.ss58_address)
        last_update = meta.last_update[idx]
        current_block = await st.get_current_block()
        
        logger.info(f"{log_prefix} verification: last_update={last_update}, ref_block={ref_block}, current_block={current_block}")
        
        if last_update >= ref_block:
            logger.info(f"{log_prefix} confirmation OK (last_update={last_update} >= ref={ref_block})")
            return True
        else:
            logger.warning(f"{log_prefix} confirmation pending (last_update={last_update} < ref={ref_block})")
            return False
            
    except ValueError:
        logger.warning(f"{log_prefix} wallet hotkey not found in metagraph")
        return False


async def _retry_async(
    func,
    *args,
    retries: int = 15,
    delay_s: float = 2.0,
    log_prefix: str = "",
    **kwargs
) -> bool:
    """Generic async retry wrapper.
    
    Args:
        func: Async function to retry
        *args: Positional arguments for func
        retries: Maximum number of retry attempts
        delay_s: Delay between retries in seconds
        log_prefix: Logging prefix
        **kwargs: Keyword arguments for func
        
    Returns:
        True if func succeeded, False if all retries exhausted
    """
    for attempt in range(retries):
        try:
            result = await func(*args, **kwargs)
            if result:
                return True
            logger.warning(f"{log_prefix} attempt {attempt+1}/{retries} returned False, retrying...")
        except Exception as e:
            logger.warning(f"{log_prefix} attempt {attempt+1}/{retries} error: {type(e).__name__}: {e}")
        
        if attempt < retries - 1:
            await asyncio.sleep(delay_s)
    
    logger.error(f"{log_prefix} failed after {retries} attempts")
    return False


async def set_weights_with_confirmation(
    wallet: "bt.wallet",
    netuid: int,
    uids: list[int],
    weights: list[float],
    wait_for_inclusion: bool = False,
    retries: int = 10,
    delay_s: float = 2.0,
    log_prefix: str = "",
    confirmation_blocks: int = 3,
) -> bool:
    """Set weights with on-chain confirmation.
    
    This function submits weights to the blockchain and waits for confirmation
    by checking the last_update field in the metagraph. It uses a retry mechanism
    to handle transient failures.
    
    Args:
        wallet: Bittensor wallet for signing
        netuid: Network UID
        uids: List of miner UIDs to set weights for
        weights: List of corresponding weights
        wait_for_inclusion: Whether to wait for block inclusion during submission
        retries: Maximum number of retry attempts
        delay_s: Delay between retries in seconds
        log_prefix: Prefix for log messages
        confirmation_blocks: Number of blocks to wait for confirmation
        
    Returns:
        True if weights were successfully set and confirmed, False otherwise
    """
    async def attempt_set_weights() -> bool:
        """Single attempt to set weights with confirmation."""
        st = await get_subtensor()
        
        # Step 1: Submit weights extrinsic
        pre_block = await st.get_current_block()
        logger.info(f"{log_prefix} submitting weights at block {pre_block}")
        
        start = time.monotonic()
        await st.set_weights(
            wallet=wallet,
            netuid=netuid,
            weights=weights,
            uids=uids,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=False,
        )
        
        submit_duration = (time.monotonic() - start) * 1000
        ref_block = await st.get_current_block()
        logger.info(f"{log_prefix} extrinsic submitted in {submit_duration:.1f}ms; ref_block={ref_block}")
        
        # Step 2: Wait for confirmation blocks
        for i in range(confirmation_blocks):
            await st.wait_for_block()
            current_block = await st.get_current_block()
            logger.info(f"{log_prefix} waited block {i+1}/{confirmation_blocks}, current_block={current_block}")
        
        # Step 3: Verify weights were updated
        return await _verify_weights_updated(
            wallet=wallet,
            netuid=netuid,
            ref_block=ref_block,
            log_prefix=log_prefix,
        )
    
    # Retry the entire process if needed
    return await _retry_async(
        attempt_set_weights,
        retries=retries,
        delay_s=delay_s,
        log_prefix=log_prefix,
    )


async def retry_set_weights(wallet: bt.Wallet, uids: List[int], weights: List[float], retry: int = 10):
    """Set weights with signer delegation or local fallback.
    
    This function attempts to delegate weight setting to a remote signer service.
    If the signer is unreachable, it falls back to local weight setting with confirmation.
    
    Args:
        wallet: Bittensor wallet for signing
        uids: List of miner UIDs
        weights: List of corresponding weights
        retry: Number of retry attempts (unused, kept for backward compatibility)
    """
    # Delegate to signer; fallback to shared helper only if signer is unreachable
    signer_url = get_conf('SIGNER_URL', default='http://signer:8080')
    try:
        logger.info(f"Calling signer at {signer_url} for set_weights uids={uids}, weights={weights}")
        parsed = urlparse(signer_url)
        try:
            infos = socket.getaddrinfo(parsed.hostname, parsed.port or 80, proto=socket.IPPROTO_TCP)
            addrs = ",".join(sorted({i[4][0] for i in infos}))
            logger.info(f"Signer DNS: host={parsed.hostname} -> {addrs}")
        except Exception as e:
            logger.warning(f"DNS resolve failed for {parsed.hostname}: {e}")
        timeout = aiohttp.ClientTimeout(connect=2, total=600)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            start = time.monotonic()
            resp = await session.post(
                f"{signer_url}/set_weights",
                json={
                    "netuid": NETUID,
                    "weights": weights,
                    "uids": uids,
                    "wait_for_inclusion": False,
                },
            )
            dur_ms = (time.monotonic() - start) * 1000
            logger.info(f"Signer HTTP response status={resp.status} in {dur_ms:.1f}ms")
            # Try to parse JSON, otherwise log text (trimmed)
            try:
                data = await resp.json()
            except Exception:
                txt = await resp.text()
                data = {"raw": (txt[:500] + ('…' if len(txt) > 500 else ''))}
            logger.info(f"Signer response body={data}")
            if resp.status == 200 and data.get("success"):
                return
            # Do not fallback if signer exists but reports failure
            logger.warning(f"Signer responded error: status={resp.status} body={data}")
            return
    except ClientConnectorError as e:
        logger.info(f"Signer not reachable ({type(e).__name__}: {e}); falling back to local set_weights once")
        ok = await set_weights_with_confirmation(
            wallet, NETUID, uids, weights, False,
            retries=int(os.getenv("SIGNER_RETRIES", "10")),
            delay_s=float(os.getenv("SIGNER_RETRY_DELAY", "2")),
            confirmation_blocks=int(os.getenv("CONFIRMATION_BLOCKS", "3")),
            log_prefix="[validator-fallback]",
        )
        if not ok:
            logger.error("Local set_weights confirmation failed")
        return
    except asyncio.TimeoutError as e:
        logger.warning(f"Signer call timed out: {e}. Not falling back to local because validator has no wallet.")
        return