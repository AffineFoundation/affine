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

import os
import sys
import click
import asyncio
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
    from affine.src.validator.main import main as validator_main, parse_args
    
    # Parse args and run
    sys.argv = ["validator"] + ctx.args
    args = parse_args()
    exit_code = asyncio.run(validator_main(args))
    sys.exit(exit_code)


# ============================================================================
# Miner Commands
# ============================================================================

@cli.command()
@click.option("--repo", required=True, help="HF repo id")
@click.option("--revision", required=True, help="HF commit SHA")
@click.option("--chute-id", required=True, help="Chutes deployment id")
@click.option("--coldkey", help="Coldkey name")
@click.option("--hotkey", help="Hotkey name")
def commit(repo, revision, chute_id, coldkey, hotkey):
    """Commit model to blockchain."""
    from affine.src.miner.commands import commit_command
    
    asyncio.run(commit_command(
        repo=repo,
        revision=revision,
        chute_id=chute_id,
        coldkey=coldkey,
        hotkey=hotkey
    ))


@cli.command()
@click.argument("uid", type=int)
@click.option("--model-path", "-p", default="./model_path", type=click.Path())
@click.option("--hf-token", help="Hugging Face API token")
def pull(uid, model_path, hf_token):
    """Pull model from Hugging Face."""
    from affine.src.miner.commands import pull_command
    
    asyncio.run(pull_command(
        uid=uid,
        model_path=model_path,
        hf_token=hf_token
    ))


@cli.command()
@click.option("--repo", required=True, help="HF repo id")
@click.option("--revision", required=True, help="HF commit SHA")
@click.option("--chutes-api-key", help="Chutes API key")
@click.option("--chute-user", help="Chutes username")
def chutes_push(repo, revision, chutes_api_key, chute_user):
    """Deploy model to Chutes."""
    from affine.src.miner.commands import chutes_push_command
    
    asyncio.run(chutes_push_command(
        repo=repo,
        revision=revision,
        chutes_api_key=chutes_api_key,
        chute_user=chute_user
    ))


@cli.command("get-sample")
@click.argument("uid", type=int)
@click.argument("env", type=str)
@click.argument("task_id", type=str)
def get_sample(uid, env, task_id):
    """Query sample result by UID, environment, and task ID.
    
    Example:
        af get-sample 42 affine task_123
    """
    from affine.src.miner.commands import get_sample_command
    
    asyncio.run(get_sample_command(
        uid=uid,
        env=env,
        task_id=task_id,
    ))


@cli.command("get-miner")
@click.argument("uid", type=int)
def get_miner(uid):
    """Query miner status and information by UID.
    
    Returns complete miner info including hotkey, model, revision,
    chute_id, validation status, and timestamps.
    
    Example:
        af get-miner 42
    """
    from affine.src.miner.commands import get_miner_command
    
    asyncio.run(get_miner_command(
        uid=uid,
    ))


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()