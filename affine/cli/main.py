#!/usr/bin/env python3
"""
Affine CLI - Unified Command Line Interface

Provides a single entry point for all Affine components:

Server Services (af servers):
- af servers api       : Start API server
- af servers executor  : Start executor service
- af servers monitor   : Start monitor service (miners monitoring)
- af servers scorer    : Start scorer service
- af servers scheduler : Start scheduler service
- af servers validator : Start validator service

Miner Commands:
- af commit      : Commit model to blockchain (miner)
- af pull        : Pull model from Hugging Face (miner)
- af chutes_push : Deploy model to Chutes (miner)
- af get-sample  : Query sample by UID, env, and task ID
- af get-miner   : Query miner status by UID
- af get-weights : Query latest normalized weights
- af get-scores  : Query latest scores for top N miners
- af get-pool    : Query pending task IDs for a miner
- af get-rank    : Query and display miner ranking table

Docker Commands:
- af deploy : Deploy docker containers (validator/backend)
- af down   : Stop docker containers (validator/backend)

Database Commands:
- af db : Database management commands
"""

import sys
import os
import subprocess
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
# Server Services (Group)
# ============================================================================

@cli.group()
def servers():
    """Start various backend server services."""
    pass


@servers.command()
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


@servers.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.pass_context
def executor(ctx):
    """Start executor service."""
    from affine.src.executor.main import main as executor_main
    
    # Forward to original executor CLI with captured -v flags
    sys.argv = ["executor"] + ctx.args
    executor_main.main(standalone_mode=False)


@servers.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.pass_context
def monitor(ctx):
    """Start monitor service."""
    from affine.src.monitor.main import main as monitor_main
    
    sys.argv = ["monitor"] + ctx.args
    monitor_main.main(standalone_mode=False)


@servers.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.pass_context
def scorer(ctx):
    """Start scorer service."""
    from affine.src.scorer.main import main as scorer_main
    
    sys.argv = ["scorer"] + ctx.args
    scorer_main.main(standalone_mode=False)


@servers.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.pass_context
def scheduler(ctx):
    """Start scheduler service."""
    from affine.src.scheduler.main import main as scheduler_main
    
    sys.argv = ["scheduler"] + ctx.args
    scheduler_main.main(standalone_mode=False)


@servers.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
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


@cli.command("get-rank", context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.pass_context
def get_rank(ctx):
    """Query and display miner ranking table.
    
    Fetches the latest score snapshot from the API and displays
    it in the same format as the scorer's detailed table output.
    
    Example:
        af get-rank
    """
    from affine.src.miner.main import get_rank as miner_get_rank
    
    sys.argv = ["get-rank"] + ctx.args
    miner_get_rank.main(standalone_mode=False)


# ============================================================================
# Database Management Commands
# ============================================================================

# Import and register the db group from database.cli
from affine.database.cli import db
cli.add_command(db)


# ============================================================================
# Docker Deployment Commands
# ============================================================================

@cli.command()
@click.argument("service", type=click.Choice(["validator", "backend"]))
@click.option("--local", is_flag=True, help="Use local build mode")
@click.option("--recreate", is_flag=True, help="Recreate containers")
def deploy(service, local, recreate):
    """Deploy docker containers for validator or backend services.
    
    SERVICE: Either 'validator' or 'backend'
    
    Examples:
        af deploy validator --recreate --local
        af deploy backend --local
        af deploy validator
        af deploy backend --recreate
    """
    # Get the affine directory (where docker-compose files are located)
    affine_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Build docker-compose command based on service type
    if service == "validator":
        compose_files = ["-f", "docker-compose.yml"]
        if local:
            compose_files.extend(["-f", "docker-compose.local.yml"])
    else:  # backend
        compose_files = ["-f", "docker-compose.backend.yml"]
        if local:
            compose_files.extend(["-f", "docker-compose.backend.local.yml"])
    
    # Build the command
    cmd = ["docker", "compose"] + compose_files + ["up", "-d"]
    
    if recreate:
        cmd.append("--force-recreate")
    
    if local:
        cmd.append("--build")
    
    # Execute the command
    logger.info(f"Deploying {service} services...")
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=affine_dir,
            check=True,
            capture_output=False
        )
        logger.info(f"✓ {service.capitalize()} services deployed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Failed to deploy {service} services")
        sys.exit(e.returncode)


@cli.command()
@click.argument("service", type=click.Choice(["validator", "backend"]))
@click.option("--local", is_flag=True, help="Use local build mode")
@click.option("--volumes", "-v", is_flag=True, help="Remove volumes as well")
def down(service, local, volumes):
    """Stop and remove docker containers for validator or backend services.
    
    SERVICE: Either 'validator' or 'backend'
    
    Examples:
        af down validator --local
        af down backend --local --volumes
        af down validator
        af down backend -v
    """
    # Get the affine directory (where docker-compose files are located)
    affine_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Build docker-compose command based on service type
    if service == "validator":
        compose_files = ["-f", "docker-compose.yml"]
        if local:
            compose_files.extend(["-f", "docker-compose.local.yml"])
    else:  # backend
        compose_files = ["-f", "docker-compose.backend.yml"]
        if local:
            compose_files.extend(["-f", "docker-compose.backend.local.yml"])
    
    # Build the command
    cmd = ["docker", "compose"] + compose_files + ["down"]
    
    if volumes:
        cmd.append("--volumes")
    
    # Execute the command
    logger.info(f"Stopping {service} services...")
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=affine_dir,
            check=True,
            capture_output=False
        )
        logger.info(f"✓ {service.capitalize()} services stopped successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Failed to stop {service} services")
        sys.exit(e.returncode)


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()