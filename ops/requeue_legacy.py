#!/usr/bin/env python
"""One-shot legacy intake: enqueue every old-mechanism model for evaluation.

The old Affine mechanism committed JSON payloads ({"model": ..., "revision":
...}) that the new affine1|repo|revision|hotkey parser ignores, so none of the
incumbent miners ever enter the duel queue. This script scans on-chain
RevealedCommitments on netuid 120, keeps the latest commitment per currently
registered hotkey, and injects them directly into the validator's
state/state.json queue (bypassing min_submission_block, which every legacy
commitment predates).

Policy is preserved: injected hotkeys are burned into seen_hotkeys and their
revisions into completed_revisions, exactly as a normal enqueue would.

Usage (STOP the validator first — it flushes in-memory state over the file):

    pm2 stop affine-validator
    python ops/requeue_legacy.py            # dry run: print what would queue
    python ops/requeue_legacy.py --apply    # write state.json
    pm2 start affine-validator
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import bittensor as bt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "affine"))

from affine import chain  # noqa: E402

NETUID = 120
STATE_PATH = ROOT / "affine" / "state" / "state.json"
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


def latest_legacy_commitments(sub) -> dict[str, tuple[int, str, str]]:
    """hotkey -> (block, repo, revision) for the newest parseable JSON payload."""
    query = sub.query_map(bt.storage.Commitments.RevealedCommitments, [NETUID])
    out: dict[str, tuple[int, str, str]] = {}
    for pair in query:
        try:
            hotkey, entries = chain._decode_commitment_pair(pair)
        except Exception:
            continue
        for block, payload in sorted(entries, key=lambda e: e[0]):
            try:
                d = json.loads(payload)
            except (json.JSONDecodeError, TypeError):
                continue
            repo = str(d.get("model", "")).strip()
            revision = str(d.get("revision", "")).strip().lower()
            if not chain.REPO_RE.match(repo) or not REVISION_RE.match(revision):
                continue
            prev = out.get(hotkey)
            if prev is None or block > prev[0]:
                out[hotkey] = (int(block), repo, revision)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write state.json")
    args = ap.parse_args()

    sub = bt.subtensor(network="finney")
    meta = sub.subnets.metagraph(NETUID)
    registered = {hk: uid for uid, hk in enumerate(meta.hotkeys)}

    legacy = latest_legacy_commitments(sub)
    print(f"legacy commitments (parseable): {len(legacy)} hotkeys")

    state = json.loads(STATE_PATH.read_text())
    seen = set(state.get("seen_hotkeys", []))
    completed = set(state.get("completed_revisions", []))
    queued_repos = {e["repo"] for e in state.get("queue", [])}
    king_hotkey = (state.get("king") or {}).get("hotkey", "")

    candidates = []
    skipped = {"unregistered": 0, "seen": 0, "revision_dup": 0, "repo_dup": 0,
               "is_king": 0}
    for hk, (block, repo, revision) in sorted(legacy.items(), key=lambda kv: kv[1][0]):
        if hk not in registered:
            skipped["unregistered"] += 1
            continue
        if hk == king_hotkey:
            skipped["is_king"] += 1
            continue
        if hk in seen:
            skipped["seen"] += 1
            continue
        if revision in completed:
            skipped["revision_dup"] += 1
            continue
        if repo in queued_repos:
            skipped["repo_dup"] += 1
            continue
        candidates.append((hk, block, repo, revision))
        queued_repos.add(repo)
        completed.add(revision)
        seen.add(hk)

    print(f"skipped: {skipped}")
    print(f"would enqueue: {len(candidates)}")
    for hk, block, repo, revision in candidates:
        print(f"  uid={registered[hk]:4d} block={block} {repo}@{revision[:12]} {hk[:12]}…")

    if not args.apply:
        print("\ndry run — pass --apply to write state.json")
        return

    counter = int(state.get("id_counter", 0))
    now = datetime.now(timezone.utc).isoformat()
    for hk, block, repo, revision in candidates:
        counter += 1
        state.setdefault("queue", []).append({
            "challenge_id": f"chal-{counter:05d}",
            "hotkey": hk, "repo": repo, "revision": revision,
            "block": block, "queued_at": now, "retry_count": 0,
        })
    state["id_counter"] = counter
    state["seen_hotkeys"] = sorted(seen)
    state["completed_revisions"] = sorted(completed)
    state.setdefault("stats", {}).setdefault("queued", 0)
    state["stats"]["queued"] += len(candidates)
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=1))
    tmp.replace(STATE_PATH)
    print(f"\nwrote {STATE_PATH}: queue now {len(state['queue'])} entries")


if __name__ == "__main__":
    main()
