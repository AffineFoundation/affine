#!/usr/bin/env python3
# --------------------------------------------------------------------------- #
#                             Imports                                         #
# --------------------------------------------------------------------------- #
from __future__ import annotations
import os
import time
import click
import socket
import asyncio
import aiohttp
import asyncio
import bittensor as bt
from urllib.parse import urlparse
from aiohttp import ClientConnectorError
from typing import Any, Dict, List, Optional, Union, Tuple, Sequence, Literal, TypeVar, Awaitable

import affine as af

async def sign_results(wallet: Any, results: List["af.Result"]) -> Tuple[str, List["af.Result"]]:
    """Sign results via external signer if available, otherwise sign locally."""
    hotkey_addr: str = ""
    try:
        signer_url = af.get_conf("SIGNER_URL", default="http://signer:8080")
        timeout = aiohttp.ClientTimeout(connect=2, total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            payloads = [str(r.challenge) for r in results]
            resp = await session.post(f"{signer_url}/sign", json={"payloads": payloads})
            if resp.status == 200:
                data = await resp.json()
                sigs = data.get("signatures") or []
                hotkey_addr = data.get("hotkey") or ""
                for r, s in zip(results, sigs):
                    r.hotkey = hotkey_addr
                    r.signature = s
                return hotkey_addr, results
    except Exception as e:
        af.logger.info(f"sink: signer unavailable, using local signing: {type(e).__name__}: {e}")
    hotkey_addr = wallet.hotkey.ss58_address
    for r in results:
        r.sign(wallet)
    return hotkey_addr, results

async def _set_weights_with_confirmation(
    wallet: "bt.wallet",
    netuid: int,
    uids: list[int],
    weights: list[float],
    wait_for_inclusion: bool = False,
    retries: int = 10,
    delay_s: float = 2.0,
    log_prefix: str = "",
) -> bool:
    for attempt in range(retries):
        try:
            st = await af.get_subtensor()
            ref = await st.get_current_block()
            af.logger.info(f"{log_prefix} set_weights attempt {attempt+1}/{retries}: netuid={netuid} uids={uids} weights={weights}")
            start = time.monotonic()
            bt.subtensor(af.get_conf('SUBTENSOR_ENDPOINT', default='finney')).set_weights(
                wallet=wallet, netuid=netuid, weights=weights, uids=uids,
                wait_for_inclusion=wait_for_inclusion,
            )
            af.logger.info(f"{log_prefix} extrinsic submitted in {(time.monotonic()-start)*1000:.1f}ms; waiting next block … (ref_block={ref})")
            await st.wait_for_block()
            meta = await st.metagraph(netuid)
            try:
                idx = meta.hotkeys.index(wallet.hotkey.ss58_address)
                lu = meta.last_update[idx]
                af.logger.info(f"{log_prefix} last_update={lu}, ref_block={ref}")
                if lu >= ref:
                    af.logger.info(f"{log_prefix} confirmation OK (last_update >= ref)")
                    return True
                af.logger.warning(f"{log_prefix} confirmation not yet included (last_update < ref), retrying …")
            except ValueError:
                af.logger.warning(f"{log_prefix} wallet hotkey not found in metagraph hotkeys; retrying …")
        except Exception as e:
            af.logger.warning(f"{log_prefix} set_weights attempt {attempt+1}/{retries} error: {type(e).__name__}: {e}")
        await asyncio.sleep(delay_s)
    return False

async def retry_set_weights( wallet: bt.Wallet, uids: List[int], weights: List[float], retry: int = 10 ):
    # Delegate to signer; fallback to shared helper only if signer is unreachable
    signer_url = af.get_conf('SIGNER_URL', default='http://signer:8080')
    try:
        af.logger.info(f"Calling signer at {signer_url} for set_weights uids={uids}")
        parsed = urlparse(signer_url)
        try:
            infos = socket.getaddrinfo(parsed.hostname, parsed.port or 80, proto=socket.IPPROTO_TCP)
            addrs = ",".join(sorted({i[4][0] for i in infos}))
            af.logger.info(f"Signer DNS: host={parsed.hostname} -> {addrs}")
        except Exception as e:
            af.logger.warning(f"DNS resolve failed for {parsed.hostname}: {e}")
        timeout = aiohttp.ClientTimeout(connect=2, total=120)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            start = time.monotonic()
            resp = await session.post(
                f"{signer_url}/set_weights",
                json={
                    "netuid": af.NETUID,
                    "weights": weights,
                    "uids": uids,
                    "wait_for_inclusion": False,
                },
            )
            dur_ms = (time.monotonic() - start) * 1000
            af.logger.info(f"Signer HTTP response status={resp.status} in {dur_ms:.1f}ms")
            # Try to parse JSON, otherwise log text (trimmed)
            try:
                data = await resp.json()
            except Exception:
                txt = await resp.text()
                data = {"raw": (txt[:500] + ('…' if len(txt) > 500 else ''))}
            af.logger.info(f"Signer response body={data}")
            if resp.status == 200 and data.get("success"):
                af.LASTSET.set(time.time())
                return
            # Do not fallback if signer exists but reports failure
            af.logger.warning(f"Signer responded error: status={resp.status} body={data}")
            return
    except ClientConnectorError as e:
        af.logger.info(f"Signer not reachable ({type(e).__name__}: {e}); falling back to local set_weights once")
        ok = await _set_weights_with_confirmation(
            wallet, af.NETUID, uids, weights, False,
            retries=int(os.getenv("SIGNER_RETRIES", "10")),
            delay_s=float(os.getenv("SIGNER_RETRY_DELAY", "2")),
            log_prefix="[validator-fallback]",
        )
        if ok:
            af.LASTSET.set(time.time())
        else:
            af.logger.error("Local set_weights confirmation failed")
        return
    except asyncio.TimeoutError as e:
        af.logger.warning(f"Signer call timed out: {e}. Not falling back to local because validator has no wallet.")
        return

@af.cli.command("signer")
@click.option('--host', default=os.getenv('SIGNER_HOST', '0.0.0.0'))
@click.option('--port', default=int(os.getenv('SIGNER_PORT', '8080')))
def signer(host: str, port: int):
    """Start lightweight HTTP signer service."""
    async def _run():
        from aiohttp import web
        coldkey = af.get_conf("BT_WALLET_COLD", "default")
        hotkey = af.get_conf("BT_WALLET_HOT", "default")
        wallet = bt.wallet(name=coldkey, hotkey=hotkey)
        @web.middleware
        async def access_log(request: "web.Request", handler):
            start = time.monotonic()
            try:
                resp = await handler(request)
                return resp
            finally:
                dur = (time.monotonic() - start) * 1000
                af.logger.info(
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
                af.logger.error(f"[signer] /sign error: {e}")
                return web.json_response({"success": False, "error": str(e)}, status=500)


        async def set_weights_handler(request: "web.Request"):
            try:
                af.logger.info("[signer] /set_weights: request received")
                payload = await request.json()
                netuid = int(payload.get('netuid', af.NETUID))
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
                af.logger.info(f"[signer] /set_weights: confirmation={'ok' if ok else 'failed'}")
                return web.json_response({"success": True} if ok else {"success": False, "error": "confirmation failed"}, status=200 if ok else 500)
            except Exception as e:
                af.logger.error(f"[signer] set_weights error: {e}")
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
        af.logger.info(f"Signer service listening on http://{host}:{port} hostname={hn} ip={ip}")
        while True:
            await asyncio.sleep(3600)

    asyncio.run(_run())