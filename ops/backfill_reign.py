#!/usr/bin/env python
"""Backfill king.previous stubs with repo/revision from on-chain commitments.

The reign table shows hotkey-only "prior" rows because those entries were
seeded from the old-mechanism incentive snapshot (ops/legacy_weights.py)
without model metadata. This script fills repo/revision/block from the
latest parseable RevealedCommitment per hotkey, and stamps live UIDs from
the metagraph when registered.

S* cannot be invented for pre-S* winners — score stays null.

Usage (STOP the validator first — it flushes in-memory state over the file):

    pm2 stop affine-validator
    python ops/backfill_reign.py            # dry run
    python ops/backfill_reign.py --apply    # write state.json
    pm2 start affine-validator
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import bittensor as bt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "affine"))

from affine import chain  # noqa: E402

NETUID = 120
STATE_PATH = ROOT / "affine" / "state" / "state.json"
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


def _parse_payload(payload: str) -> tuple[str, str] | None:
    try:
        d = json.loads(payload)
        repo = str(d.get("model", "")).strip()
        revision = str(d.get("revision", "")).strip().lower()
        if chain.REPO_RE.match(repo) and REVISION_RE.match(revision):
            return repo, revision
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass
    for prefix in ("affine1", "affine"):
        try:
            repo, revision, _author = chain.parse_reveal(prefix, payload)
            return repo, revision
        except (ValueError, AttributeError):
            continue
    return None


def commitments_for(sub, hotkeys: set[str]) -> dict[str, tuple[int, str, str]]:
    """hotkey → (block, repo, revision) newest parseable commitment."""
    query = sub.query_map(bt.storage.Commitments.RevealedCommitments, [NETUID])
    out: dict[str, tuple[int, str, str]] = {}
    for pair in query:
        try:
            hotkey, entries = chain._decode_commitment_pair(pair)
        except Exception:
            continue
        if hotkey not in hotkeys:
            continue
        for block, payload in entries:
            parsed = _parse_payload(payload)
            if not parsed:
                continue
            repo, revision = parsed
            prev = out.get(hotkey)
            if prev is None or int(block) > prev[0]:
                out[hotkey] = (int(block), repo, revision)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    state = json.loads(STATE_PATH.read_text())
    king = state.get("king")
    if not isinstance(king, dict):
        sys.exit("no king in state.json")
    previous = list(king.get("previous") or [])
    if not previous:
        print("king.previous is empty — nothing to backfill")
        return

    want = {p.get("hotkey") for p in previous if p.get("hotkey")}
    print(f"looking up {len(want)} prior hotkeys on chain…")
    sub = bt.subtensor(network="finney")
    meta = sub.subnets.metagraph(NETUID)
    uid_of = {hk: int(uid) for uid, hk in enumerate(meta.hotkeys)}
    found = commitments_for(sub, want)

    changed = 0
    for i, p in enumerate(previous):
        hk = p.get("hotkey") or ""
        before = dict(p)
        hit = found.get(hk)
        if hit:
            block, repo, revision = hit
            if not p.get("repo"):
                p["repo"] = repo
            if not p.get("revision"):
                p["revision"] = revision
            if not p.get("block"):
                p["block"] = block
        uid = uid_of.get(hk)
        if uid is not None:
            p["uid"] = uid
        # No synthetic reign_number — these are pre-S* legacy winners, not
        # Affine crowns. UI shows "prior"; S* stays null.
        if p != before:
            changed += 1
        print(
            f"  [{i}] uid={p.get('uid', '—')!s:>4} "
            f"{(p.get('repo') or '—')} "
            f"@{(p.get('revision') or '')[:12] or '—'} "
            f"{hk[:16]}…"
            f"{'' if hit else '  (no commitment found)'}"
        )

    # Genesis seed has empty hotkey by design; nothing to invent.
    kh = king.get("hotkey") or ""
    if not kh:
        print("king hotkey is empty (genesis seed) — leaving as-is")
    else:
        uid = uid_of.get(kh)
        if uid is not None and king.get("uid") != uid:
            king["uid"] = uid
            changed += 1

    print(f"\n{changed} prior rows would change")
    if not args.apply:
        print("dry run — pass --apply to write state.json")
        return

    king["previous"] = previous
    state["king"] = king
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=1))
    tmp.replace(STATE_PATH)
    print(f"wrote {STATE_PATH}")


if __name__ == "__main__":
    main()
