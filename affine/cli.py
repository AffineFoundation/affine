import os
import re
import sys
import json
import time
import click
import socket
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
from typing import Dict, List, Tuple
from affine.utils.subtensor import get_subtensor
from affine.models import BaseEnv, ContainerEnv, AgentGymContainerEnv, AffineContainerEnv, Challenge
from affine.storage import sink_enqueue, CACHE_DIR
from affine.query import run, LOG_TEMPLATE
from affine.miners import get_latest_chute_id, miners
from affine.validator import get_weights, retry_set_weights, _set_weights_with_confirmation
from affine.config import get_conf
from affine.setup import NETUID, setup_logging, logger
from aiohttp import web

HEARTBEAT = None


def _build_envs() -> List[BaseEnv]:
    spec = os.getenv("AFFINE_ENV_LIST", "").strip()
    if not spec:
        raise RuntimeError("AFFINE_ENV_LIST is required and must list envs, e.g. 'webshop,agentgym:alfworld,affine:sat'")
    envs: List[BaseEnv] = []
    for tok in [t.strip() for t in spec.split(",") if t.strip()]:
        if ":" in tok:
            prefix, name = tok.split(":", 1)
            if prefix == "agentgym":
                envs.append(AgentGymContainerEnv(env_name=name))
            elif prefix == "affine":
                envs.append(AffineContainerEnv(env_name=name))
            else:
                logger.warning(f"Unknown env prefix in AFFINE_ENV_LIST: {prefix}")
        else:
            envs.append(AgentGymContainerEnv(env_name=tok))
    if not envs:
        raise RuntimeError("AFFINE_ENV_LIST contained no supported entries.")
    return envs

async def watchdog(timeout: int = 600, sleep_div: float = 6.0):
    sleep = timeout / sleep_div
    while HEARTBEAT is None:
        await asyncio.sleep(sleep)
    while True:
        elapsed = time.monotonic() - HEARTBEAT
        if elapsed > timeout:
            logging.error(f"[WATCHDOG] Process stalled {elapsed:.0f}s — exiting process.")
            os._exit(1)
        await asyncio.sleep(sleep)

@click.group()
@click.option('-v', '--verbose', count=True, help='Increase verbosity (-v INFO, -vv DEBUG, -vvv TRACE)')
def cli(verbose):
    setup_logging(verbose)

@cli.command("runner")
def runner():
    coldkey = get_conf("BT_WALLET_COLD", "default")
    hotkey  = get_conf("BT_WALLET_HOT",  "default")
    wallet  = bt.wallet(name=coldkey, hotkey=hotkey)

    async def _run():
        subtensor = None
        envs = _build_envs()

        MAX_USES       = 30
        REFRESH_S      = 600
        SINK_BATCH     = 300
        SINK_MAX_WAIT  = 60*5

        chal_cache: Dict[str, Tuple[Challenge, int]] = {}
        last_sync = 0.0
        miners_map: Dict[int, any] = {}
        env_round: Dict[str, int] = {e.name: 0 for e in envs}
        env_inflight: Dict[str, Dict[int, asyncio.Task]] = {e.name: {} for e in envs}

        sink_q: asyncio.Queue = asyncio.Queue()

        last_status_log = 0.0
        total_requests = 0
        requests_since_last_log = 0

        def ok(res_list):
            if not res_list: return False
            r = res_list[0]
            if not getattr(r.response, "success", False): return False
            return True

        async def get_env_challenge(e: BaseEnv) -> Challenge:
            key = e.name
            chal, uses = chal_cache.get(key, (None, 0))
            if chal is None or uses >= MAX_USES:
                chal, uses = Challenge(env=e, prompt=f"{e.name} placeholder", extra={}), 0
            chal.ensure_environment_task_id()
            chal_cache[key] = (chal, uses + 1)
            return chal

        async def schedule_env_round(e: BaseEnv):
            nonlocal total_requests, requests_since_last_log
            name = e.name
            if env_inflight[name]:
                return
            chal = await get_env_challenge(e)
            env_task_id = chal.environment_task_id
            chal_task_ids = {name: env_task_id} if env_task_id is not None else None
            tasks = {}
            for m in miners_map.values():
                if not getattr(m, "model", None):
                    continue
                t = asyncio.create_task(run([chal], m, timeout=180, task_ids=chal_task_ids))
                tasks[int(m.uid)] = t
                total_requests += 1
                requests_since_last_log += 1
            env_inflight[name] = tasks

        async def ensure_subtensor():
            nonlocal subtensor
            if subtensor is None:
                subtensor = await get_subtensor()
            return subtensor

        async def subtensor_alive():
            while True:
                try:
                    st = await ensure_subtensor()
                    await st.get_current_block()
                    await asyncio.sleep(30)
                except BaseException:
                    pass


        async def refresh_miners(now):
            nonlocal last_sync, miners_map
            if (now - last_sync) >= REFRESH_S or last_sync == 0:
                st = await ensure_subtensor()
                meta = await st.metagraph(NETUID)
                miners_map = await miners(meta=meta)
                last_sync = now
                logger.debug(f"refresh: miners={len(miners_map)}")

        async def sink_worker():
            nonlocal subtensor
            batch = []
            first_put_time = None
            while True:
                try:
                    if first_put_time is None:
                        logger.debug(f"sink_worker: queue size={sink_q.qsize()}")
                        item = await sink_q.get()
                        first_put_time = time.monotonic()
                        batch.append(item)
                        while len(batch) < SINK_BATCH:
                            try:
                                more = sink_q.get_nowait()
                                batch.append(more)
                            except asyncio.QueueEmpty:
                                break
                    else:
                        remaining = SINK_MAX_WAIT - (time.monotonic() - first_put_time)
                        timeout = remaining if remaining > 0.05 else 0.05
                        try:
                            item = await asyncio.wait_for(sink_q.get(), timeout=timeout)
                            batch.append(item)
                            while len(batch) < SINK_BATCH:
                                try:
                                    more = sink_q.get_nowait()
                                    batch.append(more)
                                except asyncio.QueueEmpty:
                                    break
                        except asyncio.TimeoutError:
                            pass

                    elapsed = (time.monotonic() - first_put_time) if first_put_time is not None else 0.0
                    logger.debug(f"Until Sink: {len(batch)}/{SINK_BATCH} Time: {elapsed}/{SINK_MAX_WAIT}")
                    await asyncio.sleep(3)
                    if len(batch) >= SINK_BATCH or (batch and elapsed >= SINK_MAX_WAIT):
                        try:
                            st = await ensure_subtensor()
                            blk = await st.get_current_block()
                        except BaseException as e:
                            logger.warning(f"sink_worker: get_current_block() failed, will retry later. err={e!r}")
                            continue

                        flat = []
                        for it in batch:
                            if isinstance(it, list):
                                flat.extend(it)
                            else:
                                flat.append(it)
                        logger.debug(f"sink_worker: flushing {len(flat)} results")
                        try:
                            await sink_enqueue(wallet, blk, flat)
                        except Exception as e:
                            logger.warning(f"sink_worker: sink_enqueue() failed, will retry later. err={e!r}")
                            traceback.print_exc()
                        batch.clear()
                        first_put_time = None
                except BaseException:
                    traceback.print_exc()
                    logger.error("sink_worker: unexpected error, continuing loop")
                    await asyncio.sleep(1)

        async def main_loop():
            global HEARTBEAT
            nonlocal last_status_log, requests_since_last_log
            alive_task = asyncio.create_task(subtensor_alive())
            sink_task = asyncio.create_task(sink_worker())
            try:
                while True:
                    HEARTBEAT = now = time.monotonic()
                    await refresh_miners(now)
                    if not miners_map:
                        await asyncio.sleep(1)
                        continue

                    for e in envs:
                        if isinstance(e, ContainerEnv):
                            try:
                                await e.ensure_ready()
                            except Exception as ex:
                                logger.warning(f"ensure_ready failed for {e.name}: {ex}")

                    if now - last_status_log >= 30:
                        elapsed = now - last_status_log if last_status_log > 0 else 30
                        rps = requests_since_last_log / elapsed
                        queue_size = sink_q.qsize()
                        inflight_total = sum(len(d) for d in env_inflight.values())
                        logger.info(f"[STATUS] miners={len(miners_map)} inflight={inflight_total} queue={queue_size} req/s={rps:.1f} total_req={total_requests}")
                        last_status_log = now
                        requests_since_last_log = 0

                    for e in envs:
                        await schedule_env_round(e)

                    all_tasks = [t for d in env_inflight.values() for t in d.values()]
                    if not all_tasks:
                        await asyncio.sleep(0.2)
                        continue

                    done, _ = await asyncio.wait(all_tasks, return_when=asyncio.FIRST_COMPLETED, timeout=5.0)
                    HEARTBEAT = now = time.monotonic()
                    for t in done:
                        found = None
                        for name, d in env_inflight.items():
                            for uid, tk in d.items():
                                if tk is t:
                                    found = (name, uid)
                                    break
                            if found: break
                        name, uid = found if found else ("?", -1)
                        if name in env_inflight:
                            env_inflight[name].pop(uid, None)
                        try:
                            res_list = await t
                        except Exception as e:
                            logger.debug(f"task error env={name} uid={uid}: {e}")
                            res_list = []

                        if res_list:
                            sink_q.put_nowait(res_list)
                            for r in res_list:
                                logger.debug(
                                    LOG_TEMPLATE.format(
                                        pct=0,
                                        env=r.challenge.env.name,
                                        uid=r.miner.uid,
                                        model=(r.miner.model or "")[:50],
                                        success="RECV" if getattr(r.response, "success", False) else "NULL",
                                        score=r.evaluation.score,
                                        latency=r.response.latency_seconds,
                                    )
                                )

                        if name in env_inflight and not env_inflight[name]:
                            env_round[name] = (env_round[name] + 1) % (next((e.data_len for e in envs if e.name == name), 200) or 200)
                            e = next((e for e in envs if e.name == name), None)
                            if e is not None:
                                await schedule_env_round(e)
            except asyncio.CancelledError:
                pass
            finally:
                sink_task.cancel()
                alive_task.cancel
                with contextlib.suppress(asyncio.CancelledError):
                    await sink_task
                    await alive_task

        await main_loop()

    async def main():
        timeout = int(os.getenv("AFFINE_WATCHDOG_TIMEOUT", "900"))
        await asyncio.gather(_run(), watchdog(timeout=timeout))

    asyncio.run(main())

@cli.command("signer")
@click.option('--host', default=os.getenv('SIGNER_HOST', '0.0.0.0'))
@click.option('--port', default=int(os.getenv('SIGNER_PORT', '8080')))
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
                return web.json_response({
                    "success": True,
                    "signatures": sigs,
                    "hotkey": wallet.hotkey.ss58_address
                })
            except Exception as e:
                logger.error(f"[signer] /sign error: {e}")
                return web.json_response({"success": False, "error": str(e)}, status=500)


        async def set_weights_handler(request: "web.Request"):
            try:
                logger.info("[signer] /set_weights: request received")
                payload = await request.json()
                netuid = int(payload.get('netuid', NETUID))
                uids = payload.get('uids') or []
                weights = payload.get('weights') or []
                wait_for_inclusion = bool(payload.get('wait_for_inclusion', False))
                ok = await _set_weights_with_confirmation(
                    wallet,
                    netuid,
                    uids,
                    weights,
                    wait_for_inclusion,
                    retries=int(os.getenv("SIGNER_RETRIES", "10")),
                    delay_s=float(os.getenv("SIGNER_RETRY_DELAY", "2")),
                    log_prefix="[signer]",
                )
                logger.info(f"[signer] /set_weights: confirmation={'ok' if ok else 'failed'}")
                return web.json_response({"success": True} if ok else {"success": False, "error": "confirmation failed"}, status=200 if ok else 500)
            except Exception as e:
                logger.error(f"[signer] set_weights error: {e}")
                return web.json_response({"success": False, "error": str(e)}, status=500)

        app = web.Application(middlewares=[access_log])
        app.add_routes([
            web.get('/healthz', health),
            web.post('/set_weights', set_weights_handler),
            web.post('/sign', sign_handler),
        ])
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host=host, port=port)
        await site.start()
        try:
            hn = socket.gethostname()
            ip = socket.gethostbyname(hn)
        except Exception:
            hn, ip = ("?", "?")
        logger.info(f"Signer service listening on http://{host}:{port} hostname={hn} ip={ip}")
        while True:
            await asyncio.sleep(3600)

    asyncio.run(_run())

@cli.command("validate")
def validate():
    global HEARTBEAT
    coldkey = get_conf("BT_WALLET_COLD", "default")
    hotkey  = get_conf("BT_WALLET_HOT", "default")
    wallet  = bt.wallet(name=coldkey, hotkey=hotkey)
    async def _run():
        LAST = 0
        TEMPO = 100
        subtensor = None
        while True:
            try:
                HEARTBEAT = time.monotonic()
                if subtensor is None: subtensor = await get_subtensor()
                BLOCK = await subtensor.get_current_block()
                if BLOCK % TEMPO != 0 or BLOCK <= LAST:
                    logger.debug(f'Waiting ... {BLOCK} % {TEMPO} == {BLOCK % TEMPO} != 0')
                    await subtensor.wait_for_block()
                    continue

                force_uid0 = 0.9
                uids, weights = await get_weights(scale=0.5, burn=force_uid0)
                logger.info("Setting weights ...")
                await retry_set_weights( wallet, uids=uids, weights=weights, retry = 3)
                LAST = BLOCK

            except asyncio.CancelledError: break
            except Exception as e:
                traceback.print_exc()
                logger.info(f"Error in validator loop: {e}. Continuing ...")
                subtensor = None
                await asyncio.sleep(10)
                continue

    async def main():
        await asyncio.gather(
            _run(),
            watchdog(timeout = (60 * 20))
        )
    asyncio.run(main())

@cli.command("weights")
def weights():
    asyncio.run(get_weights())

@cli.command("pull")
@click.argument("uid", type=int)
@click.option("--model_path", "-p", default = './model_path', required=True, type=click.Path(), help="Local directory to save the model")
@click.option('--hf-token', default=None, help="Hugging Face API token (env HF_TOKEN if unset)")
def pull(uid: int, model_path: str, hf_token: str):
    hf_token     = hf_token or get_conf("HF_TOKEN")

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
@click.option('--repo', required=True, help='Existing HF repo id (e.g. <user>/<repo>)')
@click.option('--revision', required=True, help='HF commit SHA to deploy')
@click.option('--chutes-api-key', default=None, help='Chutes API key (env CHUTES_API_KEY if unset)')
def push(repo: str, revision: str, chutes_api_key: str, chute_user: str):
    """Deploy an existing HF repo+revision to Chutes and print the chute info."""
    chutes_api_key = chutes_api_key or get_conf("CHUTES_API_KEY")
    chute_user     = chute_user or get_conf("CHUTE_USER")

    async def deploy_to_chutes():
        logger.debug("Building Chute config for repo=%s revision=%s", repo, revision)
        chutes_config = textwrap.dedent(f"""
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
""")
        tmp_file = Path("tmp_chute.py")
        tmp_file.write_text(chutes_config)
        logger.debug("Wrote Chute config to %s", tmp_file)

        cmd = ["chutes", "deploy", f"{tmp_file.stem}:chute", "--accept-fee"]
        env = {**os.environ, "CHUTES_API_KEY": chutes_api_key}
        proc = await asyncio.create_subprocess_exec(
            *cmd, env=env,
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
        match = re.search(r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3})\s+\|\s+(\w+)', output)
        if match and match.group(2) == "ERROR":
            logger.debug("Chutes deploy failed with the above error log")
            raise RuntimeError("Chutes deploy failed")
        if proc.returncode != 0:
            logger.debug("Chutes deploy failed with code %d", proc.returncode)
            raise RuntimeError("Chutes deploy failed")
        tmp_file.unlink(missing_ok=True)
        logger.debug("Chute deployment successful")

    asyncio.run(deploy_to_chutes())

    chute_id   = asyncio.run(get_latest_chute_id(repo, api_key=chutes_api_key))
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
@click.option('--repo', required=True, help='HF repo id (e.g. <user>/<repo>)')
@click.option('--revision', required=True, help='HF commit SHA')
@click.option('--chute-id', required=True, help='Chutes deployment id')
@click.option('--coldkey', default=None, help='Name of the cold wallet to use.')
@click.option('--hotkey',  default=None, help='Name of the hot wallet to use.')
def commit(repo: str, revision: str, chute_id: str, coldkey: str, hotkey: str):
    """Commit repo+revision+chute_id on-chain (separate from deployment)."""
    cold = coldkey or get_conf("BT_WALLET_COLD", "default")
    hot  = hotkey  or get_conf("BT_WALLET_HOT",  "default")
    wallet = bt.wallet(name=cold, hotkey=hot)

    async def _commit():
        sub = await get_subtensor()
        data = json.dumps({"model": repo, "revision": revision, "chute_id": chute_id})
        while True:
            try:
                await sub.set_reveal_commitment(wallet=wallet, netuid=NETUID, data=data, blocks_until_reveal=1)
                break
            except MetadataError as e:
                if "SpaceLimitExceeded" in str(e):
                    await sub.wait_for_block()
                else:
                    raise

    try:
        asyncio.run(_commit())
        click.echo(json.dumps({"success": True, "repo": repo, "revision": revision, "chute_id": chute_id}))
    except Exception as e:
        logger.error("Commit failed: %s", e)
        click.echo(json.dumps({"success": False, "error": str(e)}))
