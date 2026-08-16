#!/usr/bin/env python
"""Reset SN120 king to unpaid genesis and clear the payout reign chain.

Operator directive 2026-08-15: stop miner emissions, put genesis back on the
throne, and wipe the rolling king payout window so nobody is paid.

Genesis is seeded with an empty hotkey (same as first boot). With an empty
payout chain, set_rolling_weights falls back to [subnet].burn_uid.

Usage:
    python ops/reset_genesis_unpaid.py           # dry run
    python ops/reset_genesis_unpaid.py --apply   # stop validator, mutate, restart
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "affine"))

from affine.state import King, State  # noqa: E402

STATE_DIR = ROOT / "affine" / "state"
CONTRACT_TOML = ROOT / "affine" / "affine.toml"

REASON = (
    "2026-08-15 operator: clear reign payout chain; restore unpaid genesis; "
    "emissions → burn_uid"
)


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def start_validator() -> None:
    subprocess.run(["pm2", "start", "affine-validator"], check=True)
    subprocess.run(["pm2", "restart", "affine-dash"], check=False)
    print("started affine-validator; restarted affine-dash")


def seed_cfg() -> tuple[str, str]:
    cfg = tomllib.loads(CONTRACT_TOML.read_text())
    seed = cfg["seed_king"]
    repo = seed["repo"]
    revision = seed.get("revision") or ""
    if not revision:
        raise SystemExit("seed_king.revision missing in affine.toml")
    return repo, revision


def reset_to_unpaid_genesis(state: State) -> King:
    """Replace sitting king with genesis; wipe previous (no payout members)."""
    repo, revision = seed_cfg()
    prev_reign = state.king.reign_number if state.king else -1
    reign = prev_reign + 1
    challenge_id = f"reset-genesis-unpaid-{reign}"
    # Do not call set_king(): that preserves previous kings in the payout
    # window. Write a clean seed row with empty previous.
    king = King(
        hotkey="",
        repo=repo,
        revision=revision,
        block=0,
        challenge_id=challenge_id,
        reign_number=reign,
        crowned_at=_now_iso(),
        score=None,
        previous=[],
    )
    state.king = king
    state.inaccessible_hotkeys = set()
    state._append_history({
        "event": "crowned",
        "at": _now_iso(),
        "challenge_id": challenge_id,
        "hotkey": "",
        "repo": repo,
        "revision": revision,
        "block": 0,
        "reign_number": reign,
        "score": None,
        "accepted": True,
        "reason": REASON,
        "payout": "burn",
        "previous_cleared": True,
    })
    return king


def pin_burn_weights_once() -> None:
    """Best-effort immediate burn pin (rate-limit may defer)."""
    import bittensor as bt

    cfg = tomllib.loads(CONTRACT_TOML.read_text())
    sub = cfg["subnet"]
    wallet = bt.Wallet(name=cfg["wallet"]["name"], hotkey=cfg["wallet"]["hotkey"])
    subtensor = bt.subtensor(network=sub["network"])
    burn_uid = int(sub["burn_uid"])
    version_key = int(sub["weight_version_key"])
    try:
        res = subtensor.execute(
            bt.SetWeights(
                netuid=int(sub["netuid"]),
                uids=[burn_uid],
                weights=[1.0],
                version_key=version_key,
            ),
            wallet,
        )
        res.raise_for_failure()
        print(f"weights pinned to burn uid {burn_uid} (block={res.block_hash})")
    except Exception as e:
        print(f"immediate burn set_weights deferred/failed (ok if rate-limited): {e}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--apply", action="store_true",
                    help="stop validator, reset king, pin burn, restart")
    args = ap.parse_args()

    state = State(STATE_DIR)
    state.load()
    repo, revision = seed_cfg()
    king = state.king
    print(f"sitting king: reign {king.reign_number if king else None} "
          f"{king.repo if king else None}")
    print(f"payout previous members: {len(king.previous) if king else 0}")
    print(f"in_flight: {state.in_flight.challenge_id if state.in_flight else None}")
    print(f"queue: {len(state.queue)}")
    print(f"target genesis: {repo}@{revision[:12]} (hotkey='', previous=[])")
    earners = state.king_chain_hotkeys(5) if king else []
    print(f"current earning hotkeys: {earners or '(none → burn)'}")

    if not args.apply:
        print("\ndry run — pass --apply to stop validator, clear reigns, "
              "crown unpaid genesis, pin burn weights, restart")
        return

    stop_validator()
    state = State(STATE_DIR)
    state.load()
    bak = STATE_DIR / f"state.json.bak-genesis-unpaid-{_now_stamp()}"
    shutil.copy2(STATE_DIR / "state.json", bak)
    print(f"backed up {bak}")

    # King/payout only. Do not touch queue, in_flight, or duel history —
    # operator 2026-08-15: leave every queued challenger exactly as-is.
    q_before = [e.challenge_id for e in state.queue]
    inflight_before = (state.in_flight.challenge_id if state.in_flight else None)
    new_king = reset_to_unpaid_genesis(state)
    state.flush()
    q_after = [e.challenge_id for e in state.queue]
    inflight_after = (state.in_flight.challenge_id if state.in_flight else None)
    if q_before != q_after or inflight_before != inflight_after:
        raise SystemExit(
            f"queue/in_flight mutated unexpectedly: "
            f"queue {q_before}→{q_after} in_flight {inflight_before}→{inflight_after}")
    print(f"king now: reign {new_king.reign_number} {new_king.repo} "
          f"hk={new_king.hotkey!r} previous={len(new_king.previous)}")
    print(f"earning hotkeys now: {state.king_chain_hotkeys(5) or '(none → burn)'}")
    print(f"queue unchanged ({len(q_after)}): {q_after}")
    print(f"in_flight unchanged: {inflight_after}")

    pin_burn_weights_once()
    start_validator()
    print("\ndone. Snapshot should show genesis king, empty reign_chain, "
          "weights → burn_uid.")


if __name__ == "__main__":
    main()
