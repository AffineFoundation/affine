import os
import re
import sys
import json
import time
import click
import socket
import random
import asyncio
import logging
import textwrap
import traceback
import contextlib
import bittensor as bt
from pathlib import Path
from huggingface_hub import HfApi
from bittensor.core.errors import MetadataError
from huggingface_hub import snapshot_download
from typing import Any, Dict, List, Tuple
from affine.utils.subtensor import get_subtensor
from affine.storage import sink_enqueue, CACHE_DIR, load_summary
from affine.query import query_miner
from affine import tasks as affine_tasks
from affine.miners import get_latest_chute_id, miners, get_chute
from affine.validator import (
    get_weights,
    retry_set_weights,
    _set_weights_with_confirmation,
)
from affine.config import get_conf
from affine.setup import NETUID, ENVS, setup_logging, logger
from affine.weights import weights
from aiohttp import web

HEARTBEAT = None


async def watchdog(timeout: int = 600, sleep_div: float = 6.0):
    sleep = timeout / sleep_div
    while HEARTBEAT is None:
        await asyncio.sleep(sleep)
    while True:
        elapsed = time.monotonic() - HEARTBEAT
        if elapsed > timeout:
            logging.error(
                f"[WATCHDOG] Process stalled {elapsed:.0f}s — exiting process."
            )
            os._exit(1)
        await asyncio.sleep(sleep)


@click.group()
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Increase verbosity (-v INFO, -vv DEBUG, -vvv TRACE)",
)
def cli(verbose):
    setup_logging(verbose)


@cli.command("runner")
def runner():
    coldkey = get_conf("BT_WALLET_COLD", "default")
    hotkey = get_conf("BT_WALLET_HOT", "default")
    wallet = bt.wallet(name=coldkey, hotkey=hotkey)

    async def _run():
        # Configuration constants
        MINER_REFRESH_INTERVAL = 600
        SINK_BATCH_SIZE = 300
        SINK_MAX_WAIT = 300
        STATUS_LOG_INTERVAL = 30
        MAX_CONCURRENCY = int(os.getenv("AFFINE_MAX_CONCURRENCY", "30"))

        # Shared state
        miners_map: Dict[int, any] = {}
        last_miner_sync = 0.0

        # Global concurrency control
        semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

        # Initialize SDK environments from ENVS constant
        envs = []
        for env_spec in ENVS:
            try:
                env = await affine_tasks.create_environment(env_spec)
                envs.append(env)
                logger.debug(f"Initialized environment: {env_spec}")
            except ValueError as e:
                logger.warning(f"Failed to create environment '{env_spec}': {e}")

        if not envs:
            raise RuntimeError("No valid environments initialized from ENVS")

        # Initialize subtensor
        subtensor = await get_subtensor()

        # Producer-consumer queue for storing evaluation results
        result_queue: asyncio.Queue = asyncio.Queue()

        # Metrics tracking
        total_requests_submitted = 0
        requests_since_last_status = 0
        last_status_log_time = 0.0

        async def keep_subtensor_alive():
            """Background task to keep subtensor connection alive."""
            while True:
                try:
                    await subtensor.get_current_block()
                    await asyncio.sleep(30)
                except Exception:
                    pass

        async def refresh_miners_if_needed(current_time: float):
            """Refresh miner list if refresh interval has elapsed."""
            nonlocal last_miner_sync, miners_map
            if (
                current_time - last_miner_sync
            ) >= MINER_REFRESH_INTERVAL or last_miner_sync == 0:
                meta = await subtensor.metagraph(NETUID)
                miners_map = await miners(meta=meta)
                last_miner_sync = current_time
                logger.debug(f"Refreshed miners: count={len(miners_map)}")

        async def result_sink_consumer():
            """
            Consumer: continuously drain result_queue and batch-upload to storage.
            Flushes when batch size reached or max wait time exceeded.
            """
            batch = []
            batch_start_time = None
            last_debug_log_time = 0.0

            while True:
                try:
                    # Wait for first item if batch is empty
                    if batch_start_time is None:
                        logger.debug(
                            f"Sink consumer waiting, queue_size={result_queue.qsize()}"
                        )
                        item = await result_queue.get()
                        batch_start_time = time.monotonic()
                        batch.append(item)

                        # Try to grab more items immediately
                        while len(batch) < SINK_BATCH_SIZE:
                            try:
                                batch.append(result_queue.get_nowait())
                            except asyncio.QueueEmpty:
                                break
                    else:
                        # Wait for more items with timeout
                        elapsed = time.monotonic() - batch_start_time
                        remaining = SINK_MAX_WAIT - elapsed
                        timeout = max(remaining, 0.05)

                        try:
                            item = await asyncio.wait_for(
                                result_queue.get(), timeout=timeout
                            )
                            batch.append(item)

                            # Try to grab more items
                            while len(batch) < SINK_BATCH_SIZE:
                                try:
                                    batch.append(result_queue.get_nowait())
                                except asyncio.QueueEmpty:
                                    break
                        except asyncio.TimeoutError:
                            pass

                    elapsed = (
                        (time.monotonic() - batch_start_time)
                        if batch_start_time
                        else 0.0
                    )

                    # Log batch status every 30 seconds
                    now = time.monotonic()
                    if now - last_debug_log_time >= 30.0:
                        logger.debug(
                            f"Sink batch status: {len(batch)}/{SINK_BATCH_SIZE}, elapsed={elapsed:.1f}/{SINK_MAX_WAIT}"
                        )
                        last_debug_log_time = now

                    # Flush batch if size or time threshold reached
                    if len(batch) >= SINK_BATCH_SIZE or (
                        batch and elapsed >= SINK_MAX_WAIT
                    ):
                        await asyncio.sleep(3)  # Rate limiting

                        try:
                            current_block = await subtensor.get_current_block()
                        except BaseException as e:
                            logger.warning(
                                f"Failed to get current block, retry later: {e!r}"
                            )
                            continue

                        # Flatten nested lists
                        flattened = []
                        for item in batch:
                            if isinstance(item, list):
                                flattened.extend(item)
                            else:
                                flattened.append(item)
                        logger.debug(
                            f"Flushing batch: {len(batch)}/{SINK_BATCH_SIZE}, elapsed={elapsed:.1f}/{SINK_MAX_WAIT}, flattened: {len(flattened)}, results to storage"
                        )
                        try:
                            await sink_enqueue(wallet, current_block, flattened)
                        except Exception as e:
                            logger.warning(f"Sink upload failed, will retry: {e!r}")
                            traceback.print_exc()

                        batch.clear()
                        batch_start_time = None

                except Exception:
                    traceback.print_exc()
                    logger.error("sink_worker: unexpected error, continuing loop")
                    await asyncio.sleep(1)

        async def task_producer():
            """
            Producer: continuously generate evaluation tasks for all env-miner pairs.
            Uses semaphore to control global concurrency across all environments.
            """
            global HEARTBEAT
            nonlocal total_requests_submitted, requests_since_last_status, last_status_log_time

            async def execute_with_semaphore(env, miner, task_id):
                """Wrapper to execute task with semaphore control."""
                async with semaphore:
                    return await query_miner(env, miner, task_id=task_id)

            # Track inflight tasks and pending work
            inflight_tasks: Dict[Tuple[str, int], asyncio.Task] = {}
            pending_queue: List[Tuple[Any, Any, int]] = []
            queue_index = 0
            
            while True:
                HEARTBEAT = current_time = time.monotonic()

                # Refresh miner list periodically
                await refresh_miners_if_needed(current_time)
                if not miners_map:
                    await asyncio.sleep(1)
                    continue

                # Log status periodically
                if current_time - last_status_log_time >= STATUS_LOG_INTERVAL:
                    elapsed = (
                        current_time - last_status_log_time
                        if last_status_log_time > 0
                        else STATUS_LOG_INTERVAL
                    )
                    rps = requests_since_last_status / elapsed
                    queue_size = result_queue.qsize()
                    logger.info(
                        f"[STATUS] miners={len(miners_map)} inflight={len(inflight_tasks)} "
                        f"concurrency={MAX_CONCURRENCY} queue={queue_size} req/s={rps:.1f} total={total_requests_submitted}"
                    )
                    last_status_log_time = current_time
                    requests_since_last_status = 0

                # Build pending work queue if empty (round-robin distribution)
                if not pending_queue:
                    for miner in miners_map.values():
                        if not getattr(miner, "model", None):
                            continue
                        for env in envs:
                            task_key = (env.env_name, miner.uid)
                            if task_key not in inflight_tasks:
                                data_len = getattr(env, "data_len", 1)
                                task_id = random.randint(0, data_len - 1) % data_len
                                pending_queue.append((env, miner, task_id))
                    
                    # Shuffle to randomize which miner-env pairs get submitted first
                    random.shuffle(pending_queue)
                    queue_index = 0

                # Submit tasks from pending queue up to concurrency limit
                slots_available = MAX_CONCURRENCY - len(inflight_tasks)
                while queue_index < len(pending_queue) and slots_available > 0:
                    env, miner, task_id = pending_queue[queue_index]
                    task_key = (env.env_name, miner.uid)
                    
                    # Double-check not already running (defensive)
                    if task_key not in inflight_tasks:
                        task = asyncio.create_task(
                            execute_with_semaphore(env, miner, task_id)
                        )
                        inflight_tasks[task_key] = task
                        total_requests_submitted += 1
                        requests_since_last_status += 1
                        slots_available -= 1
                    
                    queue_index += 1
                
                # Clear queue if fully processed
                if queue_index >= len(pending_queue):
                    pending_queue.clear()
                    queue_index = 0

                # Wait for at least one task to complete
                if not inflight_tasks:
                    await asyncio.sleep(0.2)
                    continue

                done, _ = await asyncio.wait(
                    inflight_tasks.values(), return_when=asyncio.FIRST_COMPLETED, timeout=5.0
                )
                HEARTBEAT = time.monotonic()

                # Process completed tasks
                for completed_task in done:
                    # Find the task key
                    task_key = None
                    for key, task in inflight_tasks.items():
                        if task is completed_task:
                            task_key = key
                            break

                    if task_key is None:
                        continue

                    # Remove from inflight
                    inflight_tasks.pop(task_key, None)
                    env_name, miner_uid = task_key

                    # Retrieve result
                    try:
                        result = await completed_task
                    except Exception as e:
                        logger.debug(f"Task failed env={env_name} uid={miner_uid}: {e}")
                        traceback.print_exc()
                        result = None

                    # Enqueue result for sink consumer (only successful responses)
                    if result:
                        if result.response.error is None:
                            result_queue.put_nowait(result)
                            logger.debug(
                                f"[RESULT] U{result.miner.uid:>3d} │ "
                                f"{(result.miner.model or '')[:50]:<50s} │ "
                                f"{result.challenge.env:<20} │ "
                                f"{'RECV':^4s} │ "
                                f"{result.evaluation.score:>6.4f} │ "
                                f"{result.response.latency_seconds:>6.3f}s"
                            )
                        else:
                            logger.debug(
                                f"[SKIP]   U{result.miner.uid:>3d} │ "
                                f"{result.challenge.env:<20} │ "
                                f"Failed response skipped: {result.response.error}"
                            )

        async def main_loop():
            """Main orchestration: run producer and consumer concurrently."""
            alive_task = asyncio.create_task(keep_subtensor_alive())
            sink_task = asyncio.create_task(result_sink_consumer())
            producer_task = asyncio.create_task(task_producer())

            try:
                await producer_task
            except asyncio.CancelledError:
                pass
            finally:
                sink_task.cancel()
                alive_task.cancel()
                producer_task.cancel()

                with contextlib.suppress(asyncio.CancelledError):
                    await sink_task
                    await alive_task
                    await producer_task

        await main_loop()

    async def main():
        timeout = int(os.getenv("AFFINE_WATCHDOG_TIMEOUT", "900"))
        await asyncio.gather(_run(), watchdog(timeout=timeout))

    asyncio.run(main())


@cli.command("signer")
@click.option("--host", default=os.getenv("SIGNER_HOST", "0.0.0.0"))
@click.option("--port", default=int(os.getenv("SIGNER_PORT", "8080")))
def signer(host: str, port: int):
    async def _run():
        coldkey = get_conf("BT_WALLET_COLD", "default")
        hotkey = get_conf("BT_WALLET_HOT", "default")
        wallet = bt.wallet(name=coldkey, hotkey=hotkey)

        @web.middleware
        async def access_log(request: "web.Request", handler):
            start = time.monotonic()
            try:
                resp = await handler(request)
                return resp
            finally:
                dur = (time.monotonic() - start) * 1000
                logger.info(
                    f"[signer] {request.remote} {request.method} {request.path} -> {getattr(request, 'response', None) and getattr(request.response, 'status', '?')} {dur:.1f}ms"
                )

        async def health(_request: "web.Request"):
            return web.json_response({"ok": True})

        async def sign_handler(request: "web.Request"):
            try:
                payload = await request.json()
                data = payload.get("payloads") or payload.get("data") or []
                if isinstance(data, str):
                    data = [data]
                sigs = [(wallet.hotkey.sign(data=d)).hex() for d in data]
                return web.json_response(
                    {
                        "success": True,
                        "signatures": sigs,
                        "hotkey": wallet.hotkey.ss58_address,
                    }
                )
            except Exception as e:
                logger.error(f"[signer] /sign error: {e}")
                return web.json_response(
                    {"success": False, "error": str(e)}, status=500
                )

        async def set_weights_handler(request: "web.Request"):
            try:
                logger.info("[signer] /set_weights: request received")
                payload = await request.json()
                netuid = int(payload.get("netuid", NETUID))
                uids = payload.get("uids") or []
                weights = payload.get("weights") or []
                wait_for_inclusion = bool(payload.get("wait_for_inclusion", False))
                ok = await _set_weights_with_confirmation(
                    wallet,
                    netuid,
                    uids,
                    weights,
                    wait_for_inclusion,
                    retries=int(os.getenv("SIGNER_RETRIES", "10")),
                    delay_s=float(os.getenv("SIGNER_RETRY_DELAY", "2")),
                    confirmation_blocks=int(os.getenv("CONFIRMATION_BLOCKS", "3")),
                    log_prefix="[signer]",
                )
                logger.info(
                    f"[signer] /set_weights: confirmation={'ok' if ok else 'failed'}"
                )
                return web.json_response(
                    (
                        {"success": True}
                        if ok
                        else {"success": False, "error": "confirmation failed"}
                    ),
                    status=200 if ok else 500,
                )
            except Exception as e:
                logger.error(f"[signer] set_weights error: {e}")
                return web.json_response(
                    {"success": False, "error": str(e)}, status=500
                )

        app = web.Application(middlewares=[access_log])
        app.add_routes(
            [
                web.get("/healthz", health),
                web.post("/set_weights", set_weights_handler),
                web.post("/sign", sign_handler),
            ]
        )
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host=host, port=port)
        await site.start()
        try:
            hn = socket.gethostname()
            ip = socket.gethostbyname(hn)
        except Exception:
            hn, ip = ("?", "?")
        logger.info(
            f"Signer service listening on http://{host}:{port} hostname={hn} ip={ip}"
        )
        while True:
            await asyncio.sleep(3600)

    asyncio.run(_run())


@cli.command("validate")
def validate():
    global HEARTBEAT
    coldkey = get_conf("BT_WALLET_COLD", "default")
    hotkey = get_conf("BT_WALLET_HOT", "default")
    wallet = bt.wallet(name=coldkey, hotkey=hotkey)

    async def _run():
        LAST = 0
        TEMPO = 100
        subtensor = None
        while True:
            try:
                HEARTBEAT = time.monotonic()
                if subtensor is None:
                    subtensor = await get_subtensor()
                BLOCK = await subtensor.get_current_block()
                if BLOCK % TEMPO != 0 or BLOCK <= LAST:
                    logger.debug(
                        f"Waiting ... {BLOCK} % {TEMPO} == {BLOCK % TEMPO} != 0"
                    )
                    await subtensor.wait_for_block()
                    continue

                force_uid0 = 0.0
                uids, weights = await get_weights(burn=force_uid0)
                logger.info("Setting weights ...")
                await retry_set_weights(wallet, uids=uids, weights=weights, retry=3)
                LAST = BLOCK

            except asyncio.CancelledError:
                break
            except Exception as e:
                traceback.print_exc()
                logger.info(f"Error in validator loop: {e}. Continuing ...")
                subtensor = None
                await asyncio.sleep(10)
                continue

    async def main():
        await asyncio.gather(_run(), watchdog(timeout=(60 * 20)))

    asyncio.run(main())


# Register weights command from weights module
cli.add_command(weights)


@cli.command("pull")
@click.argument("uid", type=int)
@click.option(
    "--model_path",
    "-p",
    default="./model_path",
    required=True,
    type=click.Path(),
    help="Local directory to save the model",
)
@click.option(
    "--hf-token", default=None, help="Hugging Face API token (env HF_TOKEN if unset)"
)
def pull(uid: int, model_path: str, hf_token: str):
    hf_token = hf_token or get_conf("HF_TOKEN")

    miner_map = asyncio.run(miners(uids=uid))
    miner = miner_map.get(uid)

    if miner is None:
        click.echo(f"No miner found for UID {uid}", err=True)
        sys.exit(1)
    repo_name = miner.model
    logger.info("Pulling model %s for UID %d into %s", repo_name, uid, model_path)

    try:
        snapshot_download(
            repo_id=repo_name,
            repo_type="model",
            local_dir=model_path,
            token=hf_token,
            resume_download=True,
            revision=miner.revision,
        )
        click.echo(f"Model {repo_name} pulled to {model_path}")
    except Exception as e:
        logger.error("Failed to download %s: %s", repo_name, e)
        click.echo(f"Error pulling model: {e}", err=True)
        sys.exit(1)


@cli.command("chutes_push")
@click.option("--repo", required=True, help="Existing HF repo id (e.g. <user>/<repo>)")
@click.option("--revision", required=True, help="HF commit SHA to deploy")
@click.option(
    "--chutes-api-key",
    default=None,
    help="Chutes API key (env CHUTES_API_KEY if unset)",
)
def push(repo: str, revision: str, chutes_api_key: str, chute_user: str):
    """Deploy an existing HF repo+revision to Chutes and print the chute info."""
    chutes_api_key = chutes_api_key or get_conf("CHUTES_API_KEY")
    chute_user = chute_user or get_conf("CHUTE_USER")

    async def deploy_to_chutes():
        logger.debug("Building Chute config for repo=%s revision=%s", repo, revision)
        chutes_config = textwrap.dedent(
            f"""
import os
from chutes.chute import NodeSelector
from chutes.chute.template.sglang import build_sglang_chute
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

chute = build_sglang_chute(
    username="{chute_user}",
    readme="{repo}",
    model_name="{repo}",
    image="chutes/sglang:nightly-2025081600",
    concurrency=20,
    revision="{revision}",
    node_selector=NodeSelector(
        gpu_count=1,
        include=["a100", "h100"],
    ),
    max_instances=1,
    scale_threshold=0.5,
    shutdown_after_seconds=3600,
)
"""
        )
        tmp_file = Path("tmp_chute.py")
        tmp_file.write_text(chutes_config)
        logger.debug("Wrote Chute config to %s", tmp_file)

        cmd = ["chutes", "deploy", f"{tmp_file.stem}:chute", "--accept-fee"]
        env = {**os.environ, "CHUTES_API_KEY": chutes_api_key}
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.PIPE,
        )
        if proc.stdin:
            proc.stdin.write(b"y\n")
            await proc.stdin.drain()
            proc.stdin.close()
        stdout, _ = await proc.communicate()
        output = stdout.decode(errors="ignore")
        logger.trace(output)
        import re

        match = re.search(
            r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3})\s+\|\s+(\w+)", output
        )
        if match and match.group(2) == "ERROR":
            logger.debug("Chutes deploy failed with the above error log")
            raise RuntimeError("Chutes deploy failed")
        if proc.returncode != 0:
            logger.debug("Chutes deploy failed with code %d", proc.returncode)
            raise RuntimeError("Chutes deploy failed")
        tmp_file.unlink(missing_ok=True)
        logger.debug("Chute deployment successful")

    asyncio.run(deploy_to_chutes())

    chute_id = asyncio.run(get_latest_chute_id(repo, api_key=chutes_api_key))
    chute_info = asyncio.run(get_chute(chute_id)) if chute_id else None
    payload = {
        "success": bool(chute_id),
        "chute_id": chute_id,
        "chute": chute_info,
        "repo": repo,
        "revision": revision,
    }
    click.echo(json.dumps(payload))


@cli.command("commit")
@click.option("--repo", required=True, help="HF repo id (e.g. <user>/<repo>)")
@click.option("--revision", required=True, help="HF commit SHA")
@click.option("--chute-id", required=True, help="Chutes deployment id")
@click.option("--coldkey", default=None, help="Name of the cold wallet to use.")
@click.option("--hotkey", default=None, help="Name of the hot wallet to use.")
def commit(repo: str, revision: str, chute_id: str, coldkey: str, hotkey: str):
    """Commit repo+revision+chute_id on-chain (separate from deployment)."""
    cold = coldkey or get_conf("BT_WALLET_COLD", "default")
    hot = hotkey or get_conf("BT_WALLET_HOT", "default")
    wallet = bt.wallet(name=cold, hotkey=hot)

    async def _commit():
        sub = await get_subtensor()
        data = json.dumps({"model": repo, "revision": revision, "chute_id": chute_id})
        while True:
            try:
                await sub.set_reveal_commitment(
                    wallet=wallet, netuid=NETUID, data=data, blocks_until_reveal=1
                )
                break
            except MetadataError as e:
                if "SpaceLimitExceeded" in str(e):
                    await sub.wait_for_block()
                else:
                    raise

    try:
        asyncio.run(_commit())
        click.echo(
            json.dumps(
                {
                    "success": True,
                    "repo": repo,
                    "revision": revision,
                    "chute_id": chute_id,
                }
            )
        )
    except Exception as e:
        logger.error("Commit failed: %s", e)
        click.echo(json.dumps({"success": False, "error": str(e)}))

