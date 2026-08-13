#!/usr/bin/env python
"""Requeue challenges that were terminally failed by eval-pod infra.

Dated 2026-08-13: the duel pod moved to an H200 host whose CUDA env broke
challenger loads (FlashInfer GDN JIT), then an OOM from the 12288 prefill
chunk, then a 180s client ReadTimeout — three infra faults that exhausted
chal-00596 (dent1s2) with `eval_infra_exhausted` before the fixes landed.
The checkpoint itself was never judged; give it a fresh slot.

Do NOT call State.enqueue(): the hotkey is already in seen_hotkeys /
intake_decided and would be skipped. Direct queue inject, new challenge id
via next_id() — same pattern as ops/requeue_shortz_fork.py.

Usage (STOP the validator first, or pass --apply which stops it):

    python ops/requeue_infra_failed.py            # dry run
    python ops/requeue_infra_failed.py --apply    # stop validator, inject
    # then: pm2 start affine-validator
"""

from __future__ import annotations

import argparse
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

# (hotkey, repo, revision) terminally failed by infra, never by their weights.
VICTIMS = [
    {
        "hotkey": "5GNQKfKWotaYNdSMGatosqfxdtbVm2MKcQwWgUo2TPcpkkkC",
        "repo": "dent1s2/Affine-5GNQKpkkkC-v1",
        "revision": "6ceabc8c561e071e3a5f2b28baaa15a3594b16ae",
        "from_challenge": "chal-00596",
    },
]


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def validator_pid() -> str | None:
    r = subprocess.run(["pm2", "pid", "affine-validator"],
                       capture_output=True, text=True)
    pid = (r.stdout or "").strip()
    if r.returncode != 0 or not pid or pid == "0":
        return None
    return pid


def stop_validator() -> None:
    if validator_pid() is None:
        print("affine-validator already stopped")
        return
    print("pm2 stop affine-validator")
    subprocess.run(["pm2", "stop", "affine-validator"], check=True)
    for _ in range(40):
        if validator_pid() is None:
            print("validator stopped")
            return
        time.sleep(1)
    raise SystemExit("validator did not stop within 40s")


def reveal_block(state: State, hotkey: str, repo: str) -> int:
    for row in reversed(state.intake):
        if (row.get("hotkey") == hotkey and row.get("repo") == repo
                and row.get("decision") == "enqueued"):
            try:
                b = int(row.get("block", 0))
            except (TypeError, ValueError):
                b = 0
            if b > 0:
                return b
    blocks = []
    for key in state.intake_decided:
        hk, _, rest = key.partition(":")
        if hk != hotkey:
            continue
        try:
            blocks.append(int(rest))
        except ValueError:
            continue
    if not blocks:
        raise SystemExit(f"no reveal block for {hotkey} {repo}")
    return max(blocks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="stop validator, inject, write state.json")
    args = ap.parse_args()

    state = State(STATE_DIR)
    state.load()
    print(f"queue={len(state.queue)} in_flight="
          f"{state.in_flight.challenge_id if state.in_flight else None}")

    queued_hk = {e.hotkey for e in state.queue}
    if state.in_flight is not None:
        queued_hk.add(state.in_flight.hotkey)

    todo = []
    for v in VICTIMS:
        block = reveal_block(state, v["hotkey"], v["repo"])
        if v["hotkey"] in queued_hk:
            print(f"  already queued/in-flight, skip: {v['repo']}")
            continue
        todo.append((v, block))
        print(f"  will inject {v['repo']}@{v['revision'][:12]} "
              f"block={block} (was {v['from_challenge']})")

    if not args.apply:
        print("\ndry run — pass --apply to stop validator and inject")
        return
    if not todo:
        print("nothing to inject")
        return

    stop_validator()
    # Reload after stop so a graceful in_flight requeue is visible.
    state = State(STATE_DIR)
    state.load()
    bak = STATE_DIR / f"state.json.bak-infra-requeue-{_now_stamp()}"
    shutil.copy2(STATE_DIR / "state.json", bak)
    print(f"backed up {bak}")

    for v, block in todo:
        entry = QueueEntry(
            challenge_id=state.next_id(),
            hotkey=v["hotkey"],
            repo=v["repo"],
            revision=v["revision"],
            block=block,
            queued_at=datetime.now(timezone.utc).isoformat(),
            retry_count=0,
        )
        state.queue.append(entry)
        state.stats["queued"] = int(state.stats.get("queued", 0)) + 1
        print(f"  injected {entry.challenge_id} {entry.repo}")
    state.flush()
    print(f"\nqueue now {len(state.queue)}")
    print("next: pm2 start affine-validator")


if __name__ == "__main__":
    main()
