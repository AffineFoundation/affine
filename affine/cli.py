"""
Enhanced CLI for the Affine framework.
Combines blockchain mining functionality with comprehensive configuration management.
"""

import os
import sys
import asyncio
import random
import logging
import click
from typing import Optional, Dict
from collections import defaultdict
from rich.console import Console
from rich.table import Table

from .core import run, miners, BaseEnv
from .config import config, get_config_value, setup_logging
from .utils import print_results, save_results as save_results_func, calculate_elo_ratings, print_elo_rankings
from .envs.COIN import COIN
from .envs.SAT import SAT
from .envs.SAT1 import SAT1
from .mining import deploy_model, DeploymentConfig

console = Console()
logger = logging.getLogger("affine")

# Environment registry
ENVS = {
    "COIN": COIN,
    "SAT": SAT,
    "SAT1": SAT1,
}

@click.group()
@click.option('-v', '--verbose', count=True,
              help='Increase verbosity (-v INFO, -vv DEBUG, -vvv TRACE)')
def cli(verbose):
    """🚀 Affine - Decentralized AI Evaluation Framework"""
    setup_logging(verbose)

@cli.command('run')
@click.argument('uids', callback=lambda _ctx, _param, val: [int(x) for x in val.split(',')])
@click.argument('env', type=click.Choice(list(ENVS), case_sensitive=False))
@click.option('-n', '--num-challenges', 'n', default=1, show_default=True, help='Number of challenges to generate')
@click.option('--timeout', default=30.0, show_default=True, help='Timeout for each API call (seconds)')
@click.option('--retries', default=3, show_default=True, help='Number of retries for failed requests')
@click.option('--backoff', default=1.0, show_default=True, help='Backoff multiplier for retries')
@click.option('--no-progress', is_flag=True, help='Disable progress bar')
@click.option('--save-results', is_flag=True, help='Save results to JSON file')
@click.option('--verbose-results', is_flag=True, help='Show detailed results including errors')
def run_command(uids, env, n, timeout, retries, backoff, no_progress, save_results, verbose_results):
    """Run N challenges in ENV against miners with UIDs.
    
    Examples:
        af run 1,2,3 COIN -n 5
        af run 42 SAT --timeout 60 --save-results
    """
    async def _run_async():
        logger.info("Running %d %s challenges against miners: %s", n, env, uids)
            
        # Generate challenges
        env_instance = ENVS[env.upper()]()
        challenges = await env_instance.many(n)
        logger.debug("Generated %d challenges", len(challenges))
            
        # Get miners
        miners_dict = await miners(uids)
        active_miners = {uid: m for uid, m in miners_dict.items() if m.model}
        logger.info("Found %d active miners out of %d total", len(active_miners), len(miners_dict))
            
        # Run evaluation
        results = await run(
            challenges=challenges, 
            miners_param=miners_dict,
            timeout=timeout,
            retries=retries,
            backoff=backoff,
            show_progress=not no_progress
        )
        
        # Display and optionally save results
        print_results(results, verbose=verbose_results)
        
        if save_results:
            save_results_func(results)
        
        return results
    
    asyncio.run(_run_async())

@cli.command('miners')
@click.argument('uids', required=False, callback=lambda _ctx, _param, val: [int(x) for x in val.split(',')] if val else None)
@click.option('--limit', default=50, help='Limit number of miners to display (default: 50, use 0 for all)')
@click.option('--active-only', is_flag=True, help='Show only miners with models')
def miners_command(uids, limit, active_only):
    """List miners and their information.
    
    Examples:
        af miners                        # Show top 50 miners
        af miners --limit 0             # Show ALL miners
        af miners 1,2,3                 # Show specific miners
        af miners --active-only         # Show only active miners
        af miners --active-only --limit 0  # Show all active miners
    """
    async def _miners_async():
        console.print("🔍 Fetching miner information...")
        miners_dict = await miners(uids, no_null=active_only)
        
        if not uids and limit > 0:  # If no specific UIDs and limit specified, limit the results
            miners_dict = dict(list(miners_dict.items())[:limit])
        
        # Display miners table
        table = Table(title="🤖 Bittensor Miners")
        table.add_column("UID", style="cyan")
        table.add_column("Hotkey", style="magenta")
        table.add_column("Model", style="green")
        table.add_column("Block", style="yellow")
        table.add_column("Status", style="blue")
        
        for uid, miner in miners_dict.items():
            status = "🟢 Active" if miner.model else "🔴 Inactive"
            hotkey_short = miner.hotkey[:8] + "..." if len(miner.hotkey) > 8 else miner.hotkey
            table.add_row(
                str(uid),
                hotkey_short,
                miner.model or "No model",
                str(miner.block) if miner.block else "Unknown",
                status
            )
        
        console.print(table)
        
        active_count = sum(1 for m in miners_dict.values() if m.model)
        console.print(f"\n📊 Summary: {active_count}/{len(miners_dict)} miners active")
    
    asyncio.run(_miners_async())

@cli.command('set')
@click.argument('key')
@click.argument('value')
def set_config(key: str, value: str):
    """Set a configuration value in ~/.affine/config.env
    
    Examples:
        af set CHUTES_API_KEY your_api_key
        af set HF_USER your_username
    """
    config_path = os.path.expanduser("~/.affine/config.env")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    # Read existing lines, filter out the key we're setting
    lines = []
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            lines = [line for line in f.readlines() if not line.strip().startswith(f"{key}=")]
    
    # Add the new key-value pair
    lines.append(f"{key}={value}\n")
    
    # Write back to file
    with open(config_path, 'w') as f:
        f.writelines(lines)
    
    logger.info("Set %s in %s", key, config_path)
    console.print(f"✅ Set {key} in {config_path}")

@cli.command('config')
@click.option('--show', is_flag=True, help='Show current configuration')
@click.option('--get', help='Get a specific config value')
def config_command(show, get):
    """Manage configuration settings.
    
    Examples:
        af config --show
        af config --get CHUTES_API_KEY
    """
    if show:
        console.print("📋 Current Configuration:")
        
        # Show .env file
        env_path = os.path.expanduser("~/.affine/config.env")
        if os.path.exists(env_path):
            console.print(f"\n🔧 Environment Variables ({env_path}):")
            with open(env_path, 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        # Hide sensitive values
                        if any(sensitive in key.lower() for sensitive in ['key', 'token', 'password']):
                            value = '*' * len(value)
                        console.print(f"  {key}: {value}")
        
        # Show config summary
        all_config = config.get_all()
        if all_config:
            console.print("\n⚙️  Configuration Summary:")
            for key, value in sorted(all_config.items()):
                if any(sensitive in key.lower() for sensitive in ['key', 'token', 'password']):
                    value = '*' * len(str(value))
                console.print(f"  {key}: {value}")
    
    elif get:
        value = config.get(get)
        if value:
            if any(sensitive in get.lower() for sensitive in ['key', 'token', 'password']):
                console.print(f"{get}: {'*' * len(value)}")
            else:
                console.print(f"{get}: {value}")
        else:
            console.print(f"❌ {get} not found")

@cli.command('deploy')
@click.argument('model_name')
@click.option('--chutes-api-key', help='Chutes API key')
@click.option('--hf-user', help='HuggingFace username')
@click.option('--hf-token', help='HuggingFace token')
@click.option('--chute-user', help='Chutes username')
@click.option('--wallet-cold', help='Bittensor coldkey')
@click.option('--wallet-hot', help='Bittensor hotkey')
@click.option('--chutes-image', help='Chutes deployment image')
@click.option('--existing-repo', help='Use existing HuggingFace repository instead of creating new one')
@click.option('--blocks-until-reveal', default=360, help='Blocks until reveal on Bittensor chain')
@click.option('--full-path', is_flag=True, help='Use model_name as full path instead of looking in model/ directory')
@click.option('--dry-run', is_flag=True, help='Show what would be deployed without actually deploying')
def deploy_command(model_name, chutes_api_key, hf_user, hf_token, chute_user, wallet_cold, wallet_hot, chutes_image, existing_repo, blocks_until_reveal, full_path, dry_run):
    """Deploy a model to the Chutes.ai platform.
    
    Examples:
        af deploy my_model_name --dry-run
        af deploy my_model_name --chutes-api-key YOUR_KEY
        af deploy my_model_name --existing-repo my-user/my-model
        af deploy /full/path/to/model --full-path
    """
    async def _deploy_async():
        # Determine the actual model path
        model_path = model_name if full_path else (
            os.path.join(os.path.dirname(__file__), "model", model_name) 
            if os.path.exists(os.path.join(os.path.dirname(__file__), "model", model_name))
            else os.path.join("model", model_name)
        )
        
        # Check if the model path exists
        if not os.path.exists(model_path):
            if full_path:
                console.print(f"❌ Model path does not exist: {model_path}")
            else:
                console.print(f"❌ Model '{model_name}' not found in model/ directory")
                console.print(f"💡 Expected path: {model_path}")
                console.print("💡 Use --full-path if you want to specify a complete path")
            sys.exit(1)
        
        if not os.path.isdir(model_path):
            console.print(f"❌ Path is not a directory: {model_path}")
            sys.exit(1)
        
        console.print(f"📁 Using model path: {model_path}")
        
        # Collect configuration from CLI, environment, or config files
        config_vars = {
            "chutes_api_key": get_config_value("CHUTES_API_KEY", chutes_api_key),
            "hf_user": get_config_value("HF_USER", hf_user),
            "hf_token": get_config_value("HF_TOKEN", hf_token),
            "chute_user": get_config_value("CHUTES_USER", chute_user),
            "wallet_cold": get_config_value("BT_COLDKEY", wallet_cold),
            "wallet_hot": get_config_value("BT_HOTKEY", wallet_hot),
            "chutes_image": get_config_value("CHUTES_IMAGE", chutes_image),
        }
        
        if dry_run:
            console.print("🔍 Dry run - showing deployment configuration:")
            table = Table(title="Deployment Configuration")
            table.add_column("Setting", style="cyan")
            table.add_column("Value", style="green")
            
            for key, value in config_vars.items():
                display_value = value if value else "❌ Not set"
                if value and any(sensitive in key for sensitive in ['key', 'token']):
                    display_value = '*' * len(value)
                table.add_row(key, display_value)
            
            table.add_row("existing_repo", existing_repo or "❌ Not set")
            table.add_row("blocks_until_reveal", str(blocks_until_reveal))
            
            console.print(table)
            console.print(f"\n📁 Model to deploy: {model_path}")
            return
        
        # Check required configurations
        required = ["chutes_api_key", "hf_user", "chute_user"]
        missing = [key for key in required if not config_vars[key]]
        
        if missing:
            console.print(f"❌ Missing required configuration: {', '.join(missing)}")
            console.print("💡 Use 'af set KEY VALUE' or provide via command line options")
            sys.exit(1)
        
        # Create deployment configuration
        deployment_config = DeploymentConfig(**config_vars)
        
        try:
            console.print("🚀 Starting deployment process...")
            
            # Deploy the model
            repo_id = await deploy_model(
                local_path=model_path,
                config=deployment_config,
                existing_repo=existing_repo,
                blocks_until_reveal=blocks_until_reveal
            )
            
            console.print(f"✅ Deployment completed successfully!")
            console.print(f"🎯 Repository: {repo_id}")
            console.print(f"🔗 Model URL: https://huggingface.co/{repo_id}")
            
        except Exception as e:
            logger.error("Deployment failed: %s", str(e))
            console.print(f"❌ Deployment failed: {str(e)}")
            sys.exit(1)
    
    asyncio.run(_deploy_async())

@cli.command('validate')
@click.option("--k-factor", "-k", default=32, show_default=True, help="ELO K-factor for rating updates")
@click.option("--challenges", "-c", default=2, show_default=True, help="Number of SAT challenges per game")
@click.option("--delay", "-d", default=5.0, show_default=True, help="Delay between rounds (seconds)")
@click.option("--rounds", "-r", default=0, help="Number of rounds to run (0 for infinite)")
@click.option("--show-elo", is_flag=True, help="Show ELO rankings after each round")
def validate_command(k_factor: int, challenges: int, delay: float, rounds: int, show_elo: bool):
    """
    Run continuous validation by pitting miners against each other.
    
    This command implements an ELO rating system where miners compete
    head-to-head on SAT challenges to determine relative skill levels.
    
    Examples:
        af validate --rounds 10 --show-elo
        af validate -k 64 -c 5 -d 10
    """
    elo_ratings: Dict[str, float] = defaultdict(lambda: 1500.0)
    
    def update_elo(miner_a, miner_b, score_a: float):
        """Update ELO ratings based on game result."""
        rating_a, rating_b = elo_ratings[miner_a.hotkey], elo_ratings[miner_b.hotkey]
        
        # Calculate expected scores
        expected_a = 1 / (1 + 10**((rating_b - rating_a) / 400))
        expected_b = 1 - expected_a
        
        # Update ratings
        elo_ratings[miner_a.hotkey] += k_factor * (score_a - expected_a)
        elo_ratings[miner_b.hotkey] += k_factor * ((1 - score_a) - expected_b)
        
        logger.debug("ELO updated: %s→%.1f, %s→%.1f", 
                    miner_a.hotkey[:8], elo_ratings[miner_a.hotkey],
                    miner_b.hotkey[:8], elo_ratings[miner_b.hotkey])
    
    async def play_game(miner_a, miner_b):
        """Play a game between two miners."""
        logger.info("Match: %s:%s vs %s:%s", 
                   miner_a.uid, miner_a.model, miner_b.uid, miner_b.model)
        
        # Generate challenges and run evaluation
        sat_env = SAT()
        game_challenges = await sat_env.many(challenges)
        results = await run(
            challenges=game_challenges, 
            miners_param=[miner_a, miner_b],
            show_progress=False
        )
        
        # Calculate scores
        score_a = sum(r.evaluation.score for r in results if r.miner.hotkey == miner_a.hotkey)
        score_b = sum(r.evaluation.score for r in results if r.miner.hotkey == miner_b.hotkey)
        
        # Determine winner and update ELO
        if score_a != score_b:
            update_elo(miner_a, miner_b, float(score_a > score_b))
            winner = miner_a if score_a > score_b else miner_b
            console.print(f"🏆 Winner: UID {winner.uid} ({score_a:.1f} vs {score_b:.1f})")
        else:
            # Tie-breaker: older miner (lower block number) wins
            update_elo(miner_a, miner_b, float(miner_a.block < miner_b.block))
            winner = miner_a if miner_a.block < miner_b.block else miner_b
            console.print(f"🤝 Tie-breaker: UID {winner.uid} (older block)")
    
    async def validation_loop():
        """Main validation loop."""
        round_count = 0
        
        console.print(f"🎯 Starting validator (K={k_factor}, challenges={challenges}, delay={delay}s)")
        console.print(f"📊 Running {'infinite' if rounds == 0 else rounds} rounds")
        
        while rounds == 0 or round_count < rounds:
            try:
                # Get active miners
                all_miners = await miners(no_null=True)
                active_miners = [m for m in all_miners.values() if m.model]
                
                if len(active_miners) < 2:
                    logger.warning("Only %d active miner(s), waiting %ds", len(active_miners), delay)
                    await asyncio.sleep(delay)
                    continue
                
                # Select two random miners
                miner_a, miner_b = random.sample(active_miners, 2)
                
                round_count += 1
                console.print(f"\n🎮 Round {round_count}: {len(active_miners)} active miners")
                
                # Play the game
                await play_game(miner_a, miner_b)
                
                # Show ELO rankings if requested
                if show_elo and round_count % 5 == 0:  # Every 5 rounds
                    console.print(f"\n📈 ELO Rankings after {round_count} rounds:")
                    print_elo_rankings(dict(elo_ratings), top_n=10)
                
                # Delay before next round
                if delay > 0:
                    await asyncio.sleep(delay)
                    
            except KeyboardInterrupt:
                console.print("\n⏹️  Validation stopped by user")
                break
            except Exception as e:
                logger.error("Error in validation round: %s", e)
                console.print(f"❌ Error: {e}")
                await asyncio.sleep(delay)
        
        # Final ELO rankings
        if elo_ratings:
            console.print(f"\n🏆 Final ELO Rankings after {round_count} rounds:")
            print_elo_rankings(dict(elo_ratings), top_n=20)
    
    asyncio.run(validation_loop())

@cli.command('environments')
def environments_command():
    """List available challenge environments."""
    table = Table(title="🎯 Available Environments")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="green")
    
    table.add_row("COIN", "Simple coin flip guessing challenge")
    table.add_row("SAT", "Boolean satisfiability problem solving")
    table.add_row("SAT1", "Advanced SAT with planted solutions and robust verification")
    
    console.print(table)
    console.print("\n💡 Use 'af run UIDS ENV' to run challenges")

if __name__ == "__main__":
    cli() 