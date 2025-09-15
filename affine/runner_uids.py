#!/usr/bin/env python3
from __future__ import annotations
import asyncio
import time
from typing import Any, Dict, List, Sequence

import bittensor as bt
import affine as af
import click


@af.cli.command("runner-uids")
@click.option("--uid", "uids_opt", type=int, multiple=True, help="Target UID (repeatable option).")
@click.option("--uids", "uids_csv", type=str, default="", help="Comma-separated list of UIDs (alternative to --uid).")
@click.option("--env", "envs_opt", type=str, multiple=True, help="Environment name to run (repeatable option). Default: all.")
@click.option("--rounds", type=int, default=1, show_default=True, help="Number of rounds to run.")
@click.option("--timeout", type=float, default=180.0, show_default=True, help="Timeout for a model request (seconds).")
@click.option("--retries", type=int, default=0, show_default=True, help="Number of HTTP retries per model request.")
@click.option("--backoff", type=float, default=1.0, show_default=True, help="Base exponential backoff between retries.")
@click.option("--sink/--no-sink", default=False, show_default=True, help="Persist results to database via sink().")
@click.option("--sleep", type=float, default=0.0, show_default=True, help="Pause between rounds (seconds).")
@click.option("--sample", type=int, default=0, show_default=True, help="Limit the number of envs sampled per round (0=all).")
@click.option("--wallet-cold", envvar="BT_WALLET_COLD", default=lambda: af.get_conf("BT_WALLET_COLD", "default"), show_default="env BT_WALLET_COLD or 'default'", help="Coldkey wallet name for signing if --sink.")
@click.option("--wallet-hot", envvar="BT_WALLET_HOT", default=lambda: af.get_conf("BT_WALLET_HOT", "default"), show_default="env BT_WALLET_HOT or 'default'", help="Hotkey wallet name for signing if --sink.")
def runner_uids(
    uids_opt: Sequence[int],
    uids_csv: str,
    envs_opt: Sequence[str],
    rounds: int,
    timeout: float,
    retries: int,
    backoff: float,
    sink: bool,
    sleep: float,
    sample: int,
    wallet_cold: str,
    wallet_hot: str,
):
    """Minimal runner targeting one or more specific UIDs.

    - Retrieves on-chain metadata via get_miners(uids)
    - Generates challenges for the env(s) and queries the miners via run()
    - Optionally persists results with sink()
    """

    # Aggregate UIDs from --uid (repeatable) and/or --uids CSV
    uids: List[int] = list(uids_opt)
    if uids_csv:
        for tok in uids_csv.split(","):
            tok = tok.strip()
            if tok:
                try:
                    uids.append(int(tok))
                except ValueError:
                    raise click.BadParameter(f"Invalid UID in --uids: {tok}")
    if not uids:
        raise click.UsageError("Please provide at least one UID via --uid or --uids.")

    # Prepare environment selection
    env_names_all = list(af.ENVS.keys())
    if envs_opt:
        unknown = [e for e in envs_opt if e not in af.ENVS]
        if unknown:
            raise click.BadParameter(f"Unknown environment(s): {unknown}. Available: {env_names_all}")
        env_names = list(envs_opt)
    else:
        env_names = env_names_all

    # Optional env sampling per round
    def choose_envs_for_round() -> List[str]:
        if sample and sample > 0:
            import random
            k = min(sample, len(env_names))
            return random.sample(env_names, k)
        return env_names

    async def _main():
        # Wallet if we need to sink
        wallet = None
        if sink:
            wallet = bt.wallet(name='test', hotkey='test')

        # Build env instances once
        env_instances: Dict[str, Any] = {name: af.ENVS[name]() for name in env_names_all}

        # Retrieve on-chain miners filtered by UIDs
        miners = await af.get_miners(uids)
        if not miners:
            af.logger.warning(f"No valid miner for UIDs {uids} (gated, missing model or missing commitment).")
            return
        af.logger.info(f"Resolved miners: {sorted(miners.keys())}")

        # Simple rounds loop
        for r in range(1, rounds + 1):
            round_start = time.monotonic()
            selected_envs = choose_envs_for_round()
            af.logger.info(f"[round {r}] envs={selected_envs} miners={len(miners)}")

            batch_results: List[af.Result] = []
            for env_name in selected_envs:
                env = env_instances[env_name]
                chal = await env.generate()
                results = await af.run(chal, miners, timeout=timeout, retries=retries, backoff=backoff)
                batch_results.extend(results)

            # Optional sink
            if sink and batch_results:
                await af.sink(wallet=wallet, results=batch_results)

            dur = time.monotonic() - round_start
            ok = sum(1 for r_ in batch_results if r_.response.success)
            fail = len(batch_results) - ok
            af.logger.info(f"[round {r}] finished in {dur:.2f}s: total={len(batch_results)} ok={ok} fail={fail}")

            if sleep > 0 and r < rounds:
                await asyncio.sleep(sleep)

    asyncio.run(_main()) 