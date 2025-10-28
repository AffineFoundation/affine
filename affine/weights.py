"""Weights command and related utilities for displaying validator weights."""

import time
import click
import asyncio
import logging
from affine.validator import get_weights
from affine.storage import load_summary
from affine.setup import logger


def _print_summary_header(block: int, schema_version: str, timestamp: int):
    """Print formatted summary header.

    Args:
        block: Block number
        schema_version: Schema version string
        timestamp: Unix timestamp
    """
    separator = "=" * 80
    time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
    print(f"\n{separator}")
    print(f"Validator Summary (Block: {block}, Schema: v{schema_version})")
    print(f"Timestamp: {time_str}")
    print(f"{separator}\n")


def _print_summary_stats(stats: dict):
    """Print summary statistics.

    Args:
        stats: Dictionary containing statistics
    """
    print("Statistics:")
    print(f"  Eligible miners: {stats.get('eligible_count', 0)}")
    print(f"  Active miners: {stats.get('active_count', 0)}")
    print(f"  Queryable miners: {stats.get('queryable_count', 0)}")
    print(f"  Total miners: {stats.get('total_miners', 0)}")


def _print_summary_table(header: list, rows: list):
    """Print summary table.

    Args:
        header: List of column names
        rows: List of row data
    """
    from tabulate import tabulate

    if header and rows:
        print(f"\n{tabulate(rows, header, tablefmt='plain')}")
    else:
        print("\nNo summary data available.")


async def _display_weights_summary(block: int = None):
    """Load and display weights summary from S3.

    Args:
        block: Optional block number to load. If None, loads latest.

    Raises:
        FileNotFoundError: If summary not found
        Exception: For other errors during loading
    """
    logger.info("Loading weights summary from S3...")
    summary = await load_summary(block)

    # Extract summary components
    schema_version = summary.get("schema_version", "unknown")
    timestamp = summary.get("timestamp", 0)
    block_num = summary.get("block", 0)
    data = summary.get("data", {})

    # Display formatted summary
    _print_summary_header(block_num, schema_version, timestamp)
    _print_summary_stats(data.get("stats", {}))

    if note := data.get("note"):
        print(f"\nNote: {note}")

    _print_summary_table(data.get("header", []), data.get("rows", []))


@click.command("weights")
@click.option(
    "-r", "--recompute",
    is_flag=True,
    default=False,
    help="Recompute weights from scratch without uploading to S3"
)
@click.option(
    "-b", "--block",
    type=int,
    default=None,
    help="Load summary from specific block (only works without -r)"
)
def weights(recompute: bool, block: int):
    """Display validator weights summary.

    By default, reads the latest computed summary from S3.
    Use -r to recompute weights from scratch (without saving to S3).
    Use -b BLOCK to load summary from a specific block.
    """
    async def run():
        if recompute:
            logger.info("Recomputing weights from scratch (not saving to S3)...")
            await get_weights(save_to_s3=False)
        else:
            try:
                await _display_weights_summary(block)
            except (FileNotFoundError, Exception) as e:
                error_type = "No cached summary found" if isinstance(e, FileNotFoundError) else "Failed to load summary"
                logger.error(f"{error_type}: {e}")
                logger.info("Run with -r flag to compute weights")

    asyncio.run(run())
