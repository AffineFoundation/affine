#!/usr/bin/env python3
"""Public timing line after Jacob's conditional go (2026-09-18 15:11 UTC)."""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

GUILD = "799672011265015819"
PUBLIC_CH = "1381987595881414656"
REPO = Path(__file__).resolve().parents[2]
TEXT = ("**wvk 22 timing, max timer added (operator, 15:18 UTC).** wvk 22 becomes effective **at the first duel "
        "boundary after the current queue (through `chal-00588`) is judged under wvk 21, and no later than the "
        "first duel boundary after 23:11 UTC** (15:11 + 8 h). At that deadline a cutoff entry still queued is judged "
        "under wvk 22; the duel in flight finishes under wvk 21 (no mid-duel abort). Projected flip ≈ **23:45 UTC**. "
        "Forward-only; reign 15 stands.")


def token() -> str:
    t = os.environ.get("DISCORD_BOT_TOKEN_ARBOS_BITTENSOR")
    if t:
        return t
    for line in (REPO / ".env").read_text().splitlines():
        if line.startswith("DISCORD_BOT_TOKEN_ARBOS_BITTENSOR="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("no token")


def main() -> int:
    req = urllib.request.Request(
        f"https://discord.com/api/v10/channels/{PUBLIC_CH}/messages",
        data=json.dumps({"content": TEXT, "allowed_mentions": {"parse": []}}).encode(),
        headers={"Authorization": f"Bot {token()}", "Content-Type": "application/json",
                 "User-Agent": "affine-notice/1.0"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        mid = json.loads(r.read())["id"]
    link = f"https://discord.com/channels/{GUILD}/{PUBLIC_CH}/{mid}"
    (Path(__file__).resolve().parent / "discord_wvk22_notice.links").open("a").write(f"public_deadline {link}\n")
    print(link)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
