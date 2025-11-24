#!/usr/bin/env python3
"""
Affine CLI - Unified Command Line Interface

Provides a single entry point for all Affine components:
- af api        : Start API server
- af executor   : Start executor service
- af monitor    : Start monitor service (miners monitoring)
- af scorer     : Start scorer service
- af scheduler  : Start scheduler service
- af validator  : Start validator service
- af commit     : Commit model to blockchain (miner)
- af pull       : Pull model from Hugging Face (miner)
- af chutes_push: Deploy model to Chutes (miner)
- af get-sample : Query sample by UID, env, and task ID
- af get-miner  : Query miner status by UID
"""

import sys
import click
from affine.core.setup import setup_logging, logger


@click.group()
@click.option(
    "-v", "--verbosity",
    count=True,
    help="Increase logging verbosity (-v=INFO, -vv=DEBUG, -vvv=TRACE)"
)
def cli(verbosity):
    """
    Affine CLI - Unified interface for all Affine components.
    
    Use -v, -vv, or -vvv for different logging levels.
    """
    # Convert count to verbosity level
    # -v -> 1, -vv -> 2, -vvv -> 3
    verbosity_level = min(verbosity, 3)
    setup_logging(verbosity_level)


# ============================================================================
# Backend Services
# ============================================================================

@cli.command()
def api():
    """Start API server."""
    from affine.api.server import app, config
    import uvicorn
    
    uvicorn.run(
        "affine.api.server:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.RELOAD,
        log_level=config.LOG_LEVEL.lower(),
        workers=1,
    )


@cli.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.pass_context
def executor(ctx):
    """Start executor service."""
    from affine.src.executor.main import main as executor_main
    
    # Forward to original executor CLI with captured -v flags
    sys.argv = ["executor"] + ctx.args
    executor_main.main(standalone_mode=False)


@cli.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.pass_context
def monitor(ctx):
    """Start monitor service."""
    from affine.src.monitor.main import main as monitor_main
    
    sys.argv = ["monitor"] + ctx.args
    monitor_main.main(standalone_mode=False)


@cli.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.pass_context
def scorer(ctx):
    """Start scorer service."""
    from affine.src.scorer.main import main as scorer_main
    
    sys.argv = ["scorer"] + ctx.args
    scorer_main.main(standalone_mode=False)


@cli.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.pass_context
def scheduler(ctx):
    """Start scheduler service."""
    from affine.src.scheduler.main import main as scheduler_main
    
    sys.argv = ["scheduler"] + ctx.args
    scheduler_main.main(standalone_mode=False)


@cli.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.pass_context
def validator(ctx):
    """Start validator service."""
    from affine.src.validator.main import main as validator_main
    
    sys.argv = ["validator"] + ctx.args
    validator_main.main(standalone_mode=False)


# ============================================================================
# Miner Commands
# ============================================================================

@cli.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.pass_context
def commit(ctx):
    """Commit model to blockchain."""
    from affine.src.miner.main import commit as miner_commit
    
    sys.argv = ["commit"] + ctx.args
    miner_commit.main(standalone_mode=False)


@cli.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.pass_context
def pull(ctx):
    """Pull model from Hugging Face."""
    from affine.src.miner.main import pull as miner_pull
    
    sys.argv = ["pull"] + ctx.args
    miner_pull.main(standalone_mode=False)


@cli.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.pass_context
def chutes_push(ctx):
    """Deploy model to Chutes."""
    from affine.src.miner.main import chutes_push as miner_chutes_push
    
    sys.argv = ["chutes_push"] + ctx.args
    miner_chutes_push.main(standalone_mode=False)


@cli.command("get-sample", context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.pass_context
def get_sample(ctx):
    """Query sample result by UID, environment, and task ID.
    
    Example:
        af get-sample 42 affine task_123
    """
    from affine.src.miner.main import get_sample as miner_get_sample
    
    sys.argv = ["get-sample"] + ctx.args
    miner_get_sample.main(standalone_mode=False)


@cli.command("get-miner", context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.pass_context
def get_miner(ctx):
    """Query miner status and information by UID.
    
    Returns complete miner info including hotkey, model, revision,
    chute_id, validation status, and timestamps.
    
    Example:
        af get-miner 42
    """
    from affine.src.miner.main import get_miner as miner_get_miner
    
    sys.argv = ["get-miner"] + ctx.args
    miner_get_miner.main(standalone_mode=False)


@cli.command("get-weights", context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.pass_context
def get_weights(ctx):
    """Query latest normalized weights for on-chain weight setting.
    
    Returns the most recent score snapshot with normalized weights
    for all miners, suitable for setting on-chain weights.
    
    Example:
        af get-weights
    """
    from affine.src.miner.main import get_weights as miner_get_weights
    
    sys.argv = ["get-weights"] + ctx.args
    miner_get_weights.main(standalone_mode=False)


@cli.command("get-scores", context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.pass_context
def get_scores(ctx):
    """Query latest scores for top N miners.
    
    Returns top N miners by score at the latest calculated block.
    
    Example:
        af get-scores
        af get-scores --top 10
    """
    from affine.src.miner.main import get_scores as miner_get_scores
    
    sys.argv = ["get-scores"] + ctx.args
    miner_get_scores.main(standalone_mode=False)


@cli.command("get-pool", context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.pass_context
def get_pool(ctx):
    """Query pending task IDs for a miner in an environment.
    
    Returns the list of task IDs currently in the sampling queue.
    
    Example:
        af get-pool 100 agentgym:webshop
    """
    from affine.src.miner.main import get_pool as miner_get_pool
    
    sys.argv = ["get-pool"] + ctx.args
    miner_get_pool.main(standalone_mode=False)


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()