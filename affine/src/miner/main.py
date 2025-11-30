"""
Miner CLI Commands

Provides command-line interface for miner operations:
- pull: Pull model from Hugging Face
- chutes_push: Deploy model to Chutes
- commit: Commit model to blockchain
- get-sample: Query sample by UID, environment, and task ID
- get-miner: Query miner status by UID
"""

import click
import asyncio

from affine.src.miner.commands import (
    pull_command,
    chutes_push_command,
    commit_command,
    get_sample_command,
    get_miner_command,
    get_weights_command,
    get_scores_command,
    get_score_command,
    get_pool_command,
)
from affine.src.miner.rank import get_rank_command


@click.command()
@click.argument("uid", type=int)
@click.option("--model-path", "-p", default="./model_path", type=click.Path(), help="Local directory to save the model")
@click.option("--hf-token", help="Hugging Face API token")
def pull(uid, model_path, hf_token):
    """Pull model from Hugging Face."""
    asyncio.run(pull_command(
        uid=uid,
        model_path=model_path,
        hf_token=hf_token
    ))


@click.command()
@click.option("--repo", required=True, help="HF repo id")
@click.option("--revision", required=True, help="HF commit SHA")
@click.option("--chutes-api-key", help="Chutes API key")
@click.option("--chute-user", help="Chutes username")
def chutes_push(repo, revision, chutes_api_key, chute_user):
    """Deploy model to Chutes."""
    asyncio.run(chutes_push_command(
        repo=repo,
        revision=revision,
        chutes_api_key=chutes_api_key,
        chute_user=chute_user
    ))


@click.command()
@click.option("--repo", required=True, help="HF repo id")
@click.option("--revision", required=True, help="HF commit SHA")
@click.option("--chute-id", required=True, help="Chutes deployment id")
@click.option("--coldkey", help="Coldkey name")
@click.option("--hotkey", help="Hotkey name")
def commit(repo, revision, chute_id, coldkey, hotkey):
    """Commit model to blockchain."""
    asyncio.run(commit_command(
        repo=repo,
        revision=revision,
        chute_id=chute_id,
        coldkey=coldkey,
        hotkey=hotkey
    ))


@click.command("get-sample")
@click.argument("uid", type=int)
@click.argument("env", type=str)
@click.argument("task_id", type=str)
def get_sample(uid, env, task_id):
    """Query sample result by UID, environment, and task ID.
    
    Example:
        af get-sample 42 affine task_123
    """
    asyncio.run(get_sample_command(
        uid=uid,
        env=env,
        task_id=task_id,
    ))

@click.command("get-miner")
@click.argument("uid", type=int)
def get_miner(uid):
    """Query miner status and information by UID.
    
    Returns complete miner info including hotkey, model, revision,
    chute_id, validation status, and timestamps.
    
    Example:
        af get-miner 42
    """
    asyncio.run(get_miner_command(
        uid=uid,
    ))


@click.command("get-weights")
def get_weights():
    """Query latest normalized weights for on-chain weight setting.
    
    Returns the most recent score snapshot with normalized weights
    for all miners, suitable for setting on-chain weights.
    
    Example:
        af get-weights
    """
    asyncio.run(get_weights_command())

@click.command("get-scores")
@click.option("--top", "-t", default=10, type=int, help="Return top N miners by score (default: 256)")
def get_scores(top):
    """Query latest scores for top N miners.
    
    Returns top N miners by score at the latest calculated block.
    
    Example:
        af get-scores
        af get-scores --top 10
    """
    asyncio.run(get_scores_command(top=top))

@click.command("get-score")
@click.argument("uid", type=int)
def get_score(uid):
    """Query score for a specific miner by UID.
    
    Returns the score details for the specified miner from the latest snapshot.
    
    Example:
        af get-score 42
    """
    asyncio.run(get_score_command(uid=uid))

@click.command("get-pool")
@click.argument("uid", type=int)
@click.argument("env", type=str)
@click.option("--full", is_flag=True, help="Print full task_ids lists without truncation")
def get_pool(uid, env, full):
    """Query task pool status for a miner in an environment.
    
    Returns the list of task IDs currently in the sampling queue
    for the specified miner and environment.
    
    Example:
        af get-pool 100 agentgym:webshop
        af get-pool 100 agentgym:webshop --full
    """
    asyncio.run(get_pool_command(uid=uid, env=env, full=full))


@click.command("get-rank")
def get_rank():
    """Query and display miner ranking table.
    
    Fetches the latest score snapshot from the API and displays
    it in the same format as the scorer's detailed table output.
    
    Example:
        af get-rank
    """
    asyncio.run(get_rank_command())




