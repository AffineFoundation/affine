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
)


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

