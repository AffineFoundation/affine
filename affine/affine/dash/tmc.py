"""TaoMarketCap SN detail SSE + registration-burn history for the dashboard."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx

from ..config import Config
from ..state import now_iso
from .readers import public_dir, read_json

log = logging.getLogger("affine.dash.tmc")

TMC_BASE = "https://api.taomarketcap.com"
RAO = 1_000_000_000
BACKOFF_START_S = 2.0
BACKOFF_MAX_S = 60.0
REG_HISTORY_POINTS = 400
REG_HISTORY_REFRESH_S = 30 * 60
REG_HISTORY_TIMEOUT_S = 180.0

_market: dict[str, Any] | None = None
_reg_history: dict[str, Any] | None = None
_lock = asyncio.Lock()
_reg_lock = asyncio.Lock()


def current_market() -> dict[str, Any] | None:
    return _market


def current_reg_history() -> dict[str, Any] | None:
    return _reg_history


def load_cached_market(cfg: Config) -> dict[str, Any] | None:
    data = read_json(public_dir(cfg) / "market.json")
    return data if isinstance(data, dict) else None


def load_cached_reg_history(cfg: Config) -> dict[str, Any] | None:
    data = read_json(public_dir(cfg) / "reg_history.json")
    return data if isinstance(data, dict) else None


def market_for_snapshot(cfg: Config) -> dict[str, Any] | None:
    market = current_market() or load_cached_market(cfg)
    if market is None:
        return None
    # Weights freshness: prefer the validator's own record over TMC's
    # last_commit_at — the indexer's field was observed frozen for 16h while
    # the chain kept accepting our set_weights. Both are UTC ISO strings with
    # identical offsets, so lexicographic comparison orders them correctly.
    raw = read_json(cfg.state_dir / "state.json")
    ours = raw.get("last_weights_at") if isinstance(raw, dict) else None
    if ours and str(ours) > str(market.get("weights_committed_at") or ""):
        market = {**market, "weights_committed_at": str(ours),
                  "weights_source": "validator"}
    return market


def reg_history_for_api(cfg: Config) -> dict[str, Any] | None:
    return current_reg_history() or load_cached_reg_history(cfg)


def _attach_reign_earnings(snap: dict[str, Any],
                           market: dict[str, Any]) -> dict[str, Any]:
    """Annotate reign members with estimated α/day and $/day from TMC.

    Uses equal-share weight_bps × miners_tao_per_day. Alpha is TAO share
    converted at the subnet price; USD uses TAO/USD from TMC market-data.
    Non-earners (outside the rolling payout window) get null earnings.
    """
    reign = snap.get("reign")
    if not isinstance(reign, dict):
        return snap
    members = reign.get("members")
    if not isinstance(members, list) or not members:
        return snap
    miners_tao = _as_float(market.get("miners_tao_per_day"))
    price_tao = _as_float(market.get("price_tao"))
    tao_usd = _as_float(market.get("tao_price_usd"))
    out_members = []
    for m in members:
        row = dict(m) if isinstance(m, dict) else {}
        bps = _as_int(row.get("weight_bps")) or 0
        earning = (bool(row.get("earning")) if "earning" in row else bps > 0)
        row["earning"] = earning
        if earning and miners_tao is not None and bps > 0:
            tao_day = miners_tao * (bps / 10000.0)
            alpha_day = (tao_day / price_tao) if price_tao and price_tao > 0 else None
            usd_day = (tao_day * tao_usd) if tao_usd is not None else None
            row["tao_per_day"] = round(tao_day, 4)
            row["alpha_per_day"] = (round(alpha_day, 2)
                                   if alpha_day is not None else None)
            row["usd_per_day"] = (round(usd_day, 2)
                                 if usd_day is not None else None)
        else:
            row.setdefault("tao_per_day", None)
            row.setdefault("alpha_per_day", None)
            row.setdefault("usd_per_day", None)
        out_members.append(row)
    return {**snap, "reign": {**reign, "members": out_members}}


def enrich_snapshot(cfg: Config, snap: dict[str, Any]) -> dict[str, Any]:
    """Attach live/cached TMC market fields for API + SSE consumers."""
    market = market_for_snapshot(cfg)
    if market is None:
        return snap
    out = {**snap, "market": market}
    return _attach_reign_earnings(out, market)


def _as_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _as_int(v: Any) -> int | None:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _normalize(netuid: int, payload: dict[str, Any]) -> dict[str, Any]:
    """Build the public market shape from a TMC full/delta subnet payload."""
    snap = payload.get("latest_snapshot")
    if not isinstance(snap, dict):
        snap = {}
    # Prefer nested snapshot fields; fall back to top-level (delta sometimes
    # only patches nested keys, but last_commit_at also lives at the root).
    price = _as_float(snap.get("price", payload.get("price")))
    burn = _as_float(snap.get("burn", payload.get("burn")))
    block = _as_int(snap.get("block_number", payload.get("block_number")))
    commit = (payload.get("last_commit_at")
              or snap.get("last_commit_at"))
    if commit is not None:
        commit = str(commit)
    reg = None if burn is None else burn / RAO
    miners_tao = _as_float(snap.get("miners_tao_per_day",
                                    payload.get("miners_tao_per_day")))
    return {
        "netuid": netuid,
        "price_tao": price,
        "reg_cost_tao": reg,
        "miners_tao_per_day": miners_tao,
        "weights_committed_at": commit,
        "block_number": block,
        "updated_at": now_iso(),
        "source": "tmc",
    }


def _merge_delta(base: dict[str, Any], changes: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge TMC delta `changes` into the last full payload."""
    out = dict(base)
    for k, v in changes.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            nested = dict(out[k])
            nested.update(v)
            # Nested dtao patches are also partial.
            if isinstance(v.get("dtao"), dict) and isinstance(nested.get("dtao"), dict):
                dtao = dict(nested["dtao"])
                dtao.update(v["dtao"])
                nested["dtao"] = dtao
            out[k] = nested
        else:
            out[k] = v
    return out


async def _write_json(path_name: str, cfg: Config, payload: dict[str, Any]) -> None:
    path = public_dir(cfg) / path_name
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n")
        tmp.replace(path)
    except OSError as exc:
        log.warning("%s write failed: %s", path_name, exc)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


async def _persist_market(cfg: Config, market: dict[str, Any]) -> None:
    await _write_json("market.json", cfg, market)


async def _persist_reg_history(cfg: Config, history: dict[str, Any]) -> None:
    await _write_json("reg_history.json", cfg, history)


def _point_from_row(row: dict[str, Any]) -> dict[str, Any] | None:
    burn = _as_float(row.get("burn"))
    block = _as_int(row.get("block_number"))
    ts = row.get("timestamp")
    if burn is None or block is None or ts is None:
        return None
    return {
        "t": str(ts),
        "block": block,
        "reg_tao": burn / RAO,
    }


def _downsample_burn_rows(rows: list[Any], target: int = REG_HISTORY_POINTS
                          ) -> list[dict[str, Any]]:
    """Evenly sample TMC per-block burn rows → oldest→newest chart points."""
    points: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        pt = _point_from_row(row)
        if pt is not None:
            points.append(pt)
    if not points:
        return []
    if points[0]["block"] > points[-1]["block"]:
        points.reverse()
    if len(points) <= target:
        return points
    # Always keep endpoints; sample the interior evenly.
    n = len(points)
    idxs = [0]
    for i in range(1, target - 1):
        idxs.append(round(i * (n - 1) / (target - 1)))
    idxs.append(n - 1)
    # Dedupe while preserving order (rounding can collide).
    out: list[dict[str, Any]] = []
    seen: set[int] = set()
    for i in idxs:
        if i in seen:
            continue
        seen.add(i)
        out.append(points[i])
    return out


async def _tip_reg_history(cfg: Config, market: dict[str, Any]) -> None:
    """Keep the chart tip live between full history refreshes."""
    global _reg_history
    reg = market.get("reg_cost_tao")
    block = market.get("block_number")
    if reg is None or block is None:
        return
    async with _reg_lock:
        hist = _reg_history
        if not hist or not isinstance(hist.get("points"), list):
            return
        points = list(hist["points"])
        tip = {
            "t": market.get("updated_at") or now_iso(),
            "block": int(block),
            "reg_tao": float(reg),
        }
        if points and points[-1].get("block") == tip["block"]:
            points[-1] = tip
        elif not points or int(points[-1].get("block") or 0) < tip["block"]:
            points.append(tip)
            # Bound growth between full refreshes.
            if len(points) > REG_HISTORY_POINTS + 50:
                points = _downsample_burn_rows(
                    [{"timestamp": p["t"], "block_number": p["block"],
                      "burn": p["reg_tao"] * RAO} for p in points],
                    REG_HISTORY_POINTS,
                )
        else:
            return
        hist = {
            **hist,
            "updated_at": now_iso(),
            "points": points,
            "tip_source": "sse",
        }
        _reg_history = hist
    await _persist_reg_history(cfg, hist)


async def _apply(cfg: Config, netuid: int, payload: dict[str, Any]) -> None:
    global _market
    market = _normalize(netuid, payload)
    async with _lock:
        # Keep prior fields when a sparse delta omits them.
        if _market and _market.get("netuid") == netuid:
            for key in ("price_tao", "reg_cost_tao", "miners_tao_per_day",
                        "tao_price_usd", "weights_committed_at",
                        "block_number"):
                if market.get(key) is None and _market.get(key) is not None:
                    market[key] = _market[key]
        _market = market
    await _persist_market(cfg, market)
    await _tip_reg_history(cfg, market)


async def _refresh_tao_usd(cfg: Config, client: httpx.AsyncClient) -> None:
    """Pull TAO/USD + miners_tao_per_day for reign earnings columns."""
    global _market
    headers = {
        "Authorization": cfg.secrets.taomarketcap,
        "Accept": "application/json",
    }
    md = await client.get(f"{TMC_BASE}/public/v1/market/market-data/",
                          headers=headers)
    if md.status_code != 200:
        raise RuntimeError(
            f"TMC market-data HTTP {md.status_code}: {md.text[:200]!r}")
    data = md.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"TMC market-data unexpected type: {type(data)}")
    tao_usd = _as_float(data.get("tao_price_usd")
                        or data.get("current_price")
                        or (data.get("usd_quote") or {}).get("price"))

    miners_tao = None
    sn = await client.get(
        f"{TMC_BASE}/public/v1/subnets/{cfg.netuid}/", headers=headers)
    if sn.status_code == 200:
        body = sn.json()
        if isinstance(body, dict):
            snap = body.get("latest_snapshot")
            if not isinstance(snap, dict):
                snap = {}
            miners_tao = _as_float(snap.get("miners_tao_per_day",
                                            body.get("miners_tao_per_day")))
            # Keep price/reg fresh too when SSE omitted them.
            price = _as_float(snap.get("price", body.get("price")))
        else:
            price = None
    else:
        price = None

    if tao_usd is None and miners_tao is None and price is None:
        return
    async with _lock:
        if _market is None:
            _market = {
                "netuid": cfg.netuid, "source": "tmc",
                "updated_at": now_iso(),
            }
        patch: dict[str, Any] = {"updated_at": now_iso()}
        if tao_usd is not None:
            patch["tao_price_usd"] = tao_usd
        if miners_tao is not None:
            patch["miners_tao_per_day"] = miners_tao
        if price is not None:
            patch["price_tao"] = price
        _market = {**_market, **patch}
        market = dict(_market)
    await _persist_market(cfg, market)


async def _stream_once(cfg: Config, client: httpx.AsyncClient) -> None:
    netuid = cfg.netuid
    url = f"{TMC_BASE}/public/v1/sse/subnets/{netuid}/"
    headers = {
        "Authorization": cfg.secrets.taomarketcap,
        "Accept": "text/event-stream",
    }
    last_full: dict[str, Any] = {}
    async with client.stream("GET", url, headers=headers) as resp:
        if resp.status_code != 200:
            body = (await resp.aread())[:200]
            raise RuntimeError(f"TMC SSE HTTP {resp.status_code}: {body!r}")
        event_name = "message"
        data_lines: list[str] = []
        async for raw in resp.aiter_lines():
            line = raw.rstrip("\r")
            if line == "":
                if not data_lines:
                    event_name = "message"
                    continue
                raw_data = "\n".join(data_lines)
                data_lines = []
                name = event_name
                event_name = "message"
                if name == "heartbeat" or raw_data.startswith(":"):
                    continue
                try:
                    msg = json.loads(raw_data)
                except json.JSONDecodeError:
                    continue
                if not isinstance(msg, dict):
                    continue
                kind = msg.get("type")
                if kind == "full" and isinstance(msg.get("data"), dict):
                    last_full = msg["data"]
                    await _apply(cfg, netuid, last_full)
                elif kind == "delta" and isinstance(msg.get("changes"), dict):
                    if not last_full:
                        # Wait for a full event before applying sparse deltas.
                        continue
                    last_full = _merge_delta(last_full, msg["changes"])
                    await _apply(cfg, netuid, last_full)
                elif kind == "heartbeat":
                    continue
                continue
            if line.startswith(":"):
                continue
            if line.startswith("event:"):
                event_name = line[6:].strip()
                continue
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
                continue


async def _refresh_reg_history(cfg: Config, client: httpx.AsyncClient) -> None:
    """Pull full TMC burn series, downsample, cache for the chart API."""
    global _reg_history
    netuid = cfg.netuid
    url = f"{TMC_BASE}/internal/v1/subnets/burn/{netuid}/"
    headers = {
        "Authorization": cfg.secrets.taomarketcap,
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
    }
    resp = await client.get(url, headers=headers)
    if resp.status_code != 200:
        raise RuntimeError(
            f"TMC burn history HTTP {resp.status_code}: {resp.text[:200]!r}")
    rows = resp.json()
    if not isinstance(rows, list):
        raise RuntimeError(f"TMC burn history unexpected type: {type(rows)}")
    points = _downsample_burn_rows(rows)
    if not points:
        raise RuntimeError("TMC burn history empty after downsample")
    hist = {
        "netuid": netuid,
        "source": "tmc",
        "updated_at": now_iso(),
        "raw_points": len(rows),
        "points": points,
    }
    async with _reg_lock:
        _reg_history = hist
    await _persist_reg_history(cfg, hist)
    log.info("reg history refreshed: raw=%s chart=%s first=%.4f last=%.4f τ",
             len(rows), len(points), points[0]["reg_tao"], points[-1]["reg_tao"])


async def run_market_stream(cfg: Config) -> None:
    """Long-running task: keep SN market cache fresh via TMC SSE."""
    global _market
    cached = load_cached_market(cfg)
    if cached:
        _market = cached
        log.info("loaded cached market.json (updated_at=%s)",
                 cached.get("updated_at"))

    key = cfg.secrets.taomarketcap
    if not key:
        log.warning("TAOMARKETCAP/TMC_API_KEY unset — market chips will stay empty")
        return

    backoff = BACKOFF_START_S
    # read=120s: TMC heartbeats well inside that, so a silently dead TCP
    # connection raises ReadTimeout and re-enters the reconnect loop instead
    # of stalling market data forever with nothing ever failing.
    timeout = httpx.Timeout(None, connect=30.0, read=120.0)
    while True:
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                try:
                    await _refresh_tao_usd(cfg, client)
                except Exception as exc:
                    log.warning("TMC tao_usd bootstrap failed: %s", exc)
                log.info("connecting TMC SSE netuid=%s", cfg.netuid)
                await _stream_once(cfg, client)
            log.warning("TMC SSE ended; reconnecting")
            backoff = BACKOFF_START_S
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("TMC SSE error: %s; retry in %.0fs", exc, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, BACKOFF_MAX_S)
            continue
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, BACKOFF_MAX_S)


async def run_reg_history_loop(cfg: Config) -> None:
    """Periodic full pull of TMC neuron-registration burn history."""
    global _reg_history
    cached = load_cached_reg_history(cfg)
    if cached and isinstance(cached.get("points"), list) and cached["points"]:
        _reg_history = cached
        log.info("loaded cached reg_history.json (points=%s updated_at=%s)",
                 len(cached["points"]), cached.get("updated_at"))

    key = cfg.secrets.taomarketcap
    if not key:
        return

    backoff = BACKOFF_START_S
    timeout = httpx.Timeout(REG_HISTORY_TIMEOUT_S, connect=30.0)
    while True:
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                await _refresh_reg_history(cfg, client)
                try:
                    await _refresh_tao_usd(cfg, client)
                except Exception as exc:
                    log.warning("TMC tao_usd refresh failed: %s", exc)
            backoff = BACKOFF_START_S
            await asyncio.sleep(REG_HISTORY_REFRESH_S)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("reg history refresh failed: %s; retry in %.0fs",
                        exc, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, BACKOFF_MAX_S)
