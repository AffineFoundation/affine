"""Validate by loading weights from remote R2 storage.

This module provides functionality to run a validator that downloads weights
from a remote R2 bucket and applies them to the blockchain, instead of
computing weights locally.

The weights are always downloaded from the official Affine R2 bucket:
https://pub-bf429ea7a5694b99adaf3d444cbbe64d.r2.dev/affine/weights/latest.json
"""

import json
import aiohttp
from typing import Tuple, List

from affine.setup import logger

# Official Affine R2 public URL
R2_PUBLIC_URL = "https://pub-bf429ea7a5694b99adaf3d444cbbe64d.r2.dev"
WEIGHTS_KEY = "affine/weights/latest.json"


async def download_weights_summary(timeout: int = 30) -> dict:
    """Download weights summary from R2 public URL.

    Always downloads from affine/weights/latest.json on the official R2 bucket.

    Args:
        timeout: Request timeout in seconds

    Returns:
        Dictionary containing the weights summary data

    Raises:
        Exception: If download fails or data is invalid
    """
    url = f"{R2_PUBLIC_URL}/{WEIGHTS_KEY}"
    logger.info(f"Downloading latest weights from: {url}")

    # Download the data
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
            response.raise_for_status()
            data = await response.json()

    logger.info(f"Successfully downloaded weights summary")
    logger.debug(f"Summary data keys: {list(data.keys())}")

    return data


def extract_uids_weights(summary: dict) -> Tuple[List[int], List[float]]:
    """Extract UIDs and weights from summary data.

    Args:
        summary: Dictionary containing the full summary structure
                 Expected format: {
                     'schema_version': '1.0.0',
                     'timestamp': ...,
                     'block': ...,
                     'data': {
                         'miners': {
                             'hotkey': {
                                 'uid': uid,
                                 'weight': weight,
                                 ...
                             },
                             ...
                         }
                     }
                 }

    Returns:
        Tuple of (uids, weights) lists

    Raises:
        ValueError: If summary format is invalid
    """
    # Extract data section
    if "data" not in summary:
        raise ValueError("Summary missing 'data' key")

    data = summary["data"]

    # Extract miners
    if "miners" not in data:
        raise ValueError("Summary data missing 'miners' key")

    miners_data = data["miners"]

    if not isinstance(miners_data, dict):
        raise ValueError(f"Expected 'miners' to be a dict, got {type(miners_data)}")

    # Extract UIDs and weights
    uids = []
    weights = []

    for miner_data in miners_data.values():
        if not isinstance(miner_data, dict):
            continue

        uid = miner_data.get("uid")
        weight = miner_data.get("weight")

        if uid is not None and weight is not None:
            uids.append(int(uid))
            weights.append(float(weight))

    if not uids:
        raise ValueError("No valid UIDs/weights found in summary")

    logger.info(f"Extracted {len(uids)} UIDs and weights from summary")
    logger.debug(f"UIDs: {uids[:10]}..." if len(uids) > 10 else f"UIDs: {uids}")
    logger.debug(f"Weights sum: {sum(weights):.6f}")

    return uids, weights


async def get_weights_from_r2() -> Tuple[List[int], List[float]]:
    """Download weights from R2 and extract UIDs and weights.

    Always downloads from the official Affine R2 bucket at:
    https://pub-bf429ea7a5694b99adaf3d444cbbe64d.r2.dev/affine/weights/latest.json

    Returns:
        Tuple of (uids, weights) lists
    """
    # Download and extract
    summary = await download_weights_summary()
    uids, weights = extract_uids_weights(summary)

    return uids, weights
