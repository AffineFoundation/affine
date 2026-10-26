"""Private upstream authentication; public, allowlisted live price events."""
import asyncio
import json
import logging
import math
import time
from datetime import datetime

import websockets

import httpx
from .tmc import market_for_snapshot, _refresh_tao_usd

log = logging.getLogger('affine.dash.live_prices')
_exchange_quote = None


def public_prices(market):
    if not isinstance(market, dict) or market.get('netuid') != 120:
        return None
    market = dict(market)
    usd_source = 'taomarketcap'
    exchange = _exchange_quote
    if exchange and exchange['at'] > (market.get('tao_usd_at') or 0):
        market.update(tao_price_usd=exchange['price'], tao_usd_at=exchange['at'])
        usd_source = 'kraken'
    fields = {'alpha_tao':'price_tao', 'tao_usd':'tao_price_usd',
              'alpha_tao_at':'alpha_tao_at', 'tao_usd_at':'tao_usd_at'}
    output = {'source':'taomarketcap', 'tao_usd_source':usd_source}
    for target, source in fields.items():
        value = market.get(source)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            return None
        output[target] = value
    return output


async def run_quote_refresh(cfg):
    if not cfg.secrets.taomarketcap:
        return
    delay = 1.5
    async with httpx.AsyncClient(timeout=httpx.Timeout(10, connect=5)) as client:
        while True:
            started = time.monotonic()
            wait = delay
            try:
                await _refresh_tao_usd(cfg, client)
                delay = 1.5
                wait = max(.05, delay - (time.monotonic() - started))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Never include request headers, upstream bodies or credentials.
                log.warning('Market quote refresh failed (%s)', type(exc).__name__)
                delay = min(60, delay * 2)
                wait = delay
            await asyncio.sleep(wait)


async def price_events(cfg, request):
    previous = None
    heartbeat = time.monotonic()
    while not await request.is_disconnected():
        quote = public_prices(market_for_snapshot(cfg))
        body = json.dumps(quote, sort_keys=True) if quote else None
        if body and body != previous:
            yield f'event: prices\ndata: {body}\n\n'
            previous = body
            heartbeat = time.monotonic()
        elif time.monotonic() - heartbeat >= 15:
            yield ': heartbeat\n\n'
            heartbeat = time.monotonic()
        await asyncio.sleep(.05)


def apply_exchange_trades(message):
    """Use the exchange's actual trade timestamp, never receipt time."""
    global _exchange_quote
    if not isinstance(message, dict) or message.get('channel') != 'trade':
        return
    rows = message.get('data')
    if not isinstance(rows, list):
        return
    for row in rows:
        if not isinstance(row, dict) or row.get('symbol') != 'TAO/USD':
            continue
        price = row.get('price')
        if type(price) not in (int, float) or not math.isfinite(price) or price <= 0:
            continue
        try:
            stamp = datetime.fromisoformat(str(row.get('timestamp', '')).replace('Z', '+00:00'))
            if stamp.tzinfo is None:
                continue
            at = stamp.timestamp()
        except (ValueError, OverflowError):
            continue
        if not math.isfinite(at) or at <= 0 or at > time.time() + 5:
            continue
        if _exchange_quote is None or at >= _exchange_quote['at']:
            _exchange_quote = {'price':price, 'at':at}


async def run_exchange_quotes():
    """One public TAO/USD trade subscription shared by every site visitor."""
    delay = 1
    while True:
        try:
            async with websockets.connect('wss://ws.kraken.com/v2', open_timeout=10, ping_interval=20, ping_timeout=20) as socket:
                await socket.send(json.dumps({'method':'subscribe', 'params':{'channel':'trade', 'symbol':['TAO/USD'], 'snapshot':True}}))
                while True:
                    message = json.loads(await asyncio.wait_for(socket.recv(), 30))
                    if message.get('success') is False:
                        raise RuntimeError('Exchange subscription rejected')
                    apply_exchange_trades(message)
                    if message.get('channel') == 'trade':
                        delay = 1
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning('Live USD stream reconnecting (%s)', type(exc).__name__)
            await asyncio.sleep(delay)
            delay = min(60, delay * 2)
