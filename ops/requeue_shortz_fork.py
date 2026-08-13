#!/usr/bin/env python
"""Revert cue-thought kings and requeue everyone they eclipsed.

Dated 2026-08-13 (`weight_version_key` 4→5, median thought-length floor).

Live kings legend (reign 9, empty thoughts) and thermopylae (reign 10,
`"Next command:"` cues) won by A9, not by longer reasoning. This script
drops both from the payout chain, restores guass (reign 8, real thoughts),
and injects a fresh duel slot for every model that lost a scored duel
while one of those two sat as king — plus the in-flight duel that was
mid-eval against thermo.

Do NOT call State.enqueue(): those hotkeys are already in seen_hotkeys /
intake_decided and would be skipped. Direct queue inject, new challenge
ids via next_id(). Skip the two cheat hotkeys and guass (the restored
king).

Usage (STOP the validator first, or pass --apply which stops it):

    python ops/requeue_shortz_fork.py            # dry run
    python ops/requeue_shortz_fork.py --apply    # stop validator, revert, inject
    # then push evalsrv + pm2 start affine-validator (see operator notes)
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "affine"))

from affine.state import QueueEntry, State  # noqa: E402

STATE_DIR = ROOT / "affine" / "state"
HISTORY_PATH = STATE_DIR / "history.jsonl"

LEGEND_REPO = "diceofgod/affine-5fjgc5jhxq-legend"
LEGEND_HOTKEY = "5D2m89RvkkCEPfi6xjQRQN7EmVL2bjALLy9xXi85wviY262S"
THERMO_REPO = "thermopylae-777/Affine-5eptsnvsre-v1"
THERMO_HOTKEY = "5F77Za37uCQfHmeiKHzTWa1RBa5KSLJnuJCvrQ8PSjVCCHsB"
GUASS_REPO = "ttttxxxxsada/Affine-5guassq3tu"
GUASS_HOTKEY = "5GuaSC6qTcjcfMYvUdmSEbkLo4Esmdyos4TmiyM19jisQ3TU"

CHEAT_HOTKEYS = {LEGEND_HOTKEY, THERMO_HOTKEY}
CHEAT_REPOS = {LEGEND_REPO, THERMO_REPO}
SKIP_HOTKEYS = CHEAT_HOTKEYS | {GUASS_HOTKEY}

REVERT_REASON = (
    "2026-08-13 thought-length floor (wvk=5): revert empty/cue-thought "
    "kings legend+thermopylae; restore guass"
)


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def validator_pid() -> str | None:
    r = subprocess.run(
        ["pm2", "pid", "affine-validator"],
        capture_output=True, text=True)
    pid = (r.stdout or "").strip()
    if r.returncode != 0 or not pid or pid == "0":
        return None
    return pid


def stop_validator() -> None:
    pid = validator_pid()
    if pid is None:
        print("affine-validator already stopped")
        return
    print(f"pm2 stop affine-validator (pid {pid})")
    subprocess.run(["pm2", "stop", "affine-validator"], check=True)
    for _ in range(40):
        if validator_pid() is None:
            print("validator stopped")
            return
        time.sleep(1)
    raise SystemExit("validator did not stop within 40s")


def _intake_blocks(state: State) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for key in state.intake_decided:
        if ":" not in key:
            continue
        hk, _, rest = key.partition(":")
        try:
            block = int(rest)
        except ValueError:
            continue
        out.setdefault(hk, []).append(block)
    for row in state.intake:
        hk = row.get("hotkey") or ""
        if row.get("decision") != "enqueued" or not hk:
            continue
        try:
            block = int(row.get("block", 0))
        except (TypeError, ValueError):
            continue
        if block <= 0:
            continue
        blocks = out.setdefault(hk, [])
        if block not in blocks:
            blocks.append(block)
    return out


def _block_for(hotkey: str, repo: str, blocks_by_hk: dict[str, list[int]],
               state: State) -> int:
    for row in reversed(state.intake):
        if (row.get("hotkey") == hotkey and row.get("repo") == repo
                and row.get("decision") == "enqueued"):
            try:
                b = int(row.get("block", 0))
            except (TypeError, ValueError):
                b = 0
            if b > 0:
                return b
    blocks = sorted(set(blocks_by_hk.get(hotkey, [])))
    if not blocks:
        raise SystemExit(f"no reveal block for {hotkey} {repo}")
    return blocks[-1]


def collect_eclipsed(state: State) -> list[dict]:
    """Scored losers after legend's crown, plus anyone currently queued."""
    started = False
    seen: dict[tuple[str, str], dict] = {}
    order: list[tuple[str, str]] = []
    with HISTORY_PATH.open() as fh:
        for line in fh:
            row = json.loads(line)
            if (row.get("event") == "crowned"
                    and row.get("repo") == LEGEND_REPO):
                started = True
                continue
            if not started:
                continue
            if row.get("event") != "verdict":
                continue
            hk = row.get("hotkey") or ""
            repo = row.get("repo") or ""
            rev = row.get("revision") or ""
            if not hk or hk in SKIP_HOTKEYS or repo in CHEAT_REPOS:
                continue
            key = (hk, rev)
            if key in seen:
                continue
            seen[key] = {
                "hotkey": hk, "repo": repo, "revision": rev,
                "from_challenge": row.get("challenge_id"),
                "at": row.get("at"),
            }
            order.append(key)
    blocks_by_hk = _intake_blocks(state)
    out = []
    for key in order:
        item = seen[key]
        item["block"] = _block_for(item["hotkey"], item["repo"],
                                   blocks_by_hk, state)
        out.append(item)
    queued = {(e.hotkey, e.revision) for e in state.queue}
    if state.in_flight is not None:
        e = state.in_flight
        queued.add((e.hotkey, e.revision))
        if (e.hotkey not in SKIP_HOTKEYS
                and (e.hotkey, e.revision) not in seen):
            out.append({
                "hotkey": e.hotkey, "repo": e.repo, "revision": e.revision,
                "block": e.block, "from_challenge": e.challenge_id,
                "at": e.queued_at, "in_flight": True,
            })
    for e in state.queue:
        if (e.hotkey not in SKIP_HOTKEYS
                and (e.hotkey, e.revision) not in seen
                and not any(x["hotkey"] == e.hotkey
                            and x["revision"] == e.revision for x in out)):
            out.append({
                "hotkey": e.hotkey, "repo": e.repo, "revision": e.revision,
                "block": e.block, "from_challenge": e.challenge_id,
                "at": e.queued_at, "already_queued": True,
            })
    del queued
    return out


def revert_to_guass(state: State) -> None:
    king = state.king
    if king is None:
        raise SystemExit("no sitting king")
    if king.hotkey == GUASS_HOTKEY or king.repo == GUASS_REPO:
        print(f"king already guass ({king.repo} reign {king.reign_number})")
        return
    hops = []
    if king.hotkey == THERMO_HOTKEY or king.repo == THERMO_REPO:
        hops = [LEGEND_HOTKEY, GUASS_HOTKEY]
    elif king.hotkey == LEGEND_HOTKEY or king.repo == LEGEND_REPO:
        hops = [GUASS_HOTKEY]
    else:
        raise SystemExit(
            f"sitting king is not thermo/legend/guass: {king.repo} "
            f"{king.hotkey[:16]} reign {king.reign_number}")
    for expect_hk in hops:
        restored = state.revert_king(REVERT_REASON)
        if restored is None:
            raise SystemExit("revert_king returned None")
        print(f"reverted → reign {restored.reign_number} {restored.repo}")
        if restored.hotkey != expect_hk:
            raise SystemExit(
                f"expected hotkey {expect_hk[:16]} after revert, "
                f"got {restored.hotkey[:16]} {restored.repo}")
    if state.king is None or state.king.hotkey != GUASS_HOTKEY:
        raise SystemExit("king is not guass after revert")


def inject(state: State, eclipsed: list[dict]) -> list[QueueEntry]:
    queued_hk = {e.hotkey for e in state.queue}
    if state.in_flight is not None:
        queued_hk.add(state.in_flight.hotkey)
    added: list[QueueEntry] = []
    for item in eclipsed:
        hk = item["hotkey"]
        if hk == GUASS_HOTKEY or hk in CHEAT_HOTKEYS:
            continue
        if hk in queued_hk:
            print(f"  already queued, keep as-is: {item['repo']}")
            continue
        entry = QueueEntry(
            challenge_id=state.next_id(),
            hotkey=hk,
            repo=item["repo"],
            revision=item["revision"],
            block=int(item["block"]),
            queued_at=datetime.now(timezone.utc).isoformat(),
            retry_count=0,
        )
        state.queue.append(entry)
        queued_hk.add(hk)
        state.stats["queued"] = int(state.stats.get("queued", 0)) + 1
        added.append(entry)
        print(f"  inject {entry.challenge_id} {entry.repo}@{entry.revision[:12]} "
              f"block={entry.block} (was {item.get('from_challenge')})")
    return added


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="stop validator, revert kings, write state.json")
    args = ap.parse_args()

    state = State(STATE_DIR)
    state.load()
    king = state.king
    print(f"sitting king: reign {king.reign_number if king else None} "
          f"{king.repo if king else None}")
    print(f"queue={len(state.queue)} in_flight="
          f"{state.in_flight.challenge_id if state.in_flight else None}")

    eclipsed = collect_eclipsed(state)
    print(f"eclipsed to requeue: {len(eclipsed)}")
    for item in eclipsed:
        flag = (" in_flight" if item.get("in_flight")
                else " queued" if item.get("already_queued") else "")
        print(f"  {item.get('from_challenge')} {item['repo']} "
              f"block={item['block']}{flag}")

    if not args.apply:
        print("\ndry run — pass --apply to stop validator, revert to guass, "
              "and inject the queue")
        return

    stop_validator()
    # Reload after stop so a graceful in_flight requeue is visible.
    state = State(STATE_DIR)
    state.load()
    bak = STATE_DIR / f"state.json.bak-shortz-{_now_stamp()}"
    shutil.copy2(STATE_DIR / "state.json", bak)
    print(f"backed up {bak}")

    eclipsed = collect_eclipsed(state)
    revert_to_guass(state)
    added = inject(state, eclipsed)
    state.flush()
    print(f"\nking now: reign {state.king.reign_number} {state.king.repo}")
    print(f"queue now {len(state.queue)} (injected {len(added)})")
    for e in state.queue:
        print(f"  {e.challenge_id} {e.repo}")
    print("\nnext: push evalsrv score/dueling/toml/config, restart evalsrv "
          "by its python PID (not pkill -f), then:")
    print("  pm2 start affine-validator && pm2 restart affine-dash")


if __name__ == "__main__":
    main()
