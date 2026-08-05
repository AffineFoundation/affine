"""Filesystem readers for validator state_dir (read-only)."""

from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path

from .. import __version__
from ..config import Config
from ..state import now_iso

log = logging.getLogger("affine.dash.readers")


def public_dir(cfg: Config) -> Path:
    return cfg.state_dir / "public"


def read_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("bad json %s: %s", path, exc)
        return None


def load_public_snapshot(cfg: Config) -> dict | None:
    data = read_json(public_dir(cfg) / "snapshot.json")
    return data if isinstance(data, dict) else None


def load_public_history(cfg: Config) -> list | None:
    data = read_json(public_dir(cfg) / "history.json")
    return data if isinstance(data, list) else None


def load_public_benchmarks(cfg: Config) -> dict | None:
    data = read_json(public_dir(cfg) / "benchmarks.json")
    return data if isinstance(data, dict) else None


def _reign_members(king: dict | None, depth: int) -> list[dict]:
    """Mirror State.king_chain_members without loading/mutating State."""
    if not king or depth <= 0:
        return []
    members = [{
        "reign_number": king.get("reign_number"),
        "repo": king.get("repo", ""),
        "revision": king.get("revision", ""),
        "hotkey": king.get("hotkey", ""),
        "crowned_at": king.get("crowned_at"),
        "block": king.get("block"),
        "score": king.get("score"),
        "current": True,
    }]
    seen = {king.get("hotkey", "")}
    for p in king.get("previous") or []:
        hk = p.get("hotkey", "")
        if not hk or hk in seen:
            continue
        seen.add(hk)
        members.append({
            "reign_number": p.get("reign_number"),
            "repo": p.get("repo", ""),
            "revision": p.get("revision", ""),
            "hotkey": hk,
            "crowned_at": p.get("crowned_at"),
            "block": p.get("block"),
            "score": p.get("score"),
            "current": False,
        })
        if len(members) >= depth:
            break
    members = members[:depth]
    weight_bps = 10000 // len(members)
    for m in members:
        m["weight_bps"] = weight_bps
    return members


def reconstruct_snapshot(cfg: Config) -> dict:
    """Build a snapshot from state.json when public/snapshot.json is missing.

    Read-only: never instantiates State (load() can reconcile/write).
    `current_eval` / `phase` are memory-only unless dashboard has flushed a
    public snapshot, so this fallback omits live progress.
    """
    raw = read_json(cfg.state_dir / "state.json")
    if not isinstance(raw, dict):
        raw = {}
    king = raw.get("king")
    members = _reign_members(king if isinstance(king, dict) else None,
                             cfg.king_chain_size)
    queue = raw.get("queue") or []
    return {
        "generated_at": now_iso(),
        "version": __version__,
        "phase": {"name": "unknown", "since": raw.get("flushed_at")},
        "king": ({
            "repo": king.get("repo"), "revision": king.get("revision"),
            "hotkey": king.get("hotkey"),
            "reign_number": king.get("reign_number"),
            "crowned_at": king.get("crowned_at"), "block": king.get("block"),
            "score": king.get("score"),
        } if isinstance(king, dict) else None),
        "reign": {"size": cfg.king_chain_size, "members": members},
        "reign_chain": [m["hotkey"] for m in members],
        "queue": [{
            "challenge_id": e.get("challenge_id"), "repo": e.get("repo"),
            "hotkey": e.get("hotkey"), "queued_at": e.get("queued_at"),
            "retry_count": e.get("retry_count", 0),
        } for e in queue if isinstance(e, dict)],
        "current_eval": None,
        "bench_jobs": raw.get("bench_jobs") or [],
        "stats": raw.get("stats") or {},
        "eval_machine": {
            "provider": (raw.get("eval_machine") or {}).get("provider"),
            "created_at": (raw.get("eval_machine") or {}).get("created_at"),
        } if raw.get("eval_machine") else None,
        "bench_machine": {
            "provider": (raw.get("bench_machine") or {}).get("provider"),
            "created_at": (raw.get("bench_machine") or {}).get("created_at"),
        } if raw.get("bench_machine") else None,
    }


def snapshot(cfg: Config) -> dict:
    pub = load_public_snapshot(cfg)
    if pub is not None:
        return pub
    return reconstruct_snapshot(cfg)


def contract_payload(cfg: Config) -> dict:
    return {
        "subnet": cfg.raw["subnet"],
        "submission": cfg.raw["submission"],
        "teacher": {"repo": cfg.teacher.repo},
        "dataset": cfg.raw["dataset"],
        "duel": cfg.raw["duel"],
        "bench": {k: v for k, v in cfg.raw["bench"].items()},
        "dashboard": {
            "public_base_url": cfg.dashboard.get("public_base_url", ""),
        },
        "version": __version__,
    }


def eval_artifact_path(cfg: Config, challenge_id: str) -> Path:
    return cfg.state_dir / "evals" / f"{challenge_id}.json.gz"


def load_eval_artifact(cfg: Config, challenge_id: str) -> dict | None:
    path = eval_artifact_path(cfg, challenge_id)
    if not path.exists():
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError, gzip.BadGzipFile) as exc:
        log.warning("bad eval artifact %s: %s", path, exc)
        return None


def history_row_from_raw(r: dict) -> dict:
    """Normalize a history.jsonl record into the slim dashboard shape."""
    v = r.get("verdict") or {}
    return {
        "event": r.get("event"),
        "at": r.get("at"),
        "challenge_id": r.get("challenge_id"),
        "repo": r.get("repo"),
        "hotkey": r.get("hotkey"),
        "revision": r.get("revision"),
        "accepted": r.get("accepted"),
        "error_code": r.get("error_code"),
        # Keep the validator's full failure text (state caps at 2000).
        "error_detail": (r.get("error_detail") or "")[:2000],
        "z": v.get("z"),
        "margin": v.get("margin"),
        "se": v.get("se"),
        "n_paired_turns": v.get("n_paired_turns"),
        "rejection_reason": v.get("rejection_reason"),
        "reign_number": r.get("reign_number"),
        "score": r.get("score", (v.get("challenger") or {}).get("S")),
        "score_king": (v.get("king") or {}).get("S"),
        "gates": v.get("gates"),
        "challenger": v.get("challenger"),
        "king": v.get("king"),
        "challenger_wins": v.get("challenger_wins"),
    }
