#!/usr/bin/env python3
"""One public line after reign 15 crowned under wvk 21 while the wvk-22 notice
was up (coordinator 2026-09-18 15:02 UTC). Token from the repo .env."""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

GUILD = "799672011265015819"
PUBLIC_CH = "1381987595881414656"
REPO = Path(__file__).resolve().parents[2]
TEXT = ("**Update to the wvk-22 notice.** `chal-00581` (uid 222) crowned **reign 15** at 14:59 UTC "
        "under the current rule (wvk 21, min(R, G): margin +0.0029, z 4.57 over 1,294 turns) — so the "
        "\"queue is empty\" line above is stale: duels are running and being judged under wvk 21, "
        "including a real crown. The shadow sd-meter agreed with that crown (margin +0.14 sd, z 4.7). "
        "wvk 22 flips at a **later duel boundary** once the operator gives the go; forward-only — "
        "**reign 15 stands**, no re-verdicts. Models in the queue are judged under the rule in force "
        "when their duel runs.")


def token() -> str:
    t = os.environ.get("DISCORD_BOT_TOKEN_ARBOS_BITTENSOR")
    if t:
        return t
    for line in (REPO / ".env").read_text().splitlines():
        if line.startswith("DISCORD_BOT_TOKEN_ARBOS_BITTENSOR="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("no token")


def main() -> int:
    assert len(TEXT) <= 2000
    req = urllib.request.Request(
        f"https://discord.com/api/v10/channels/{PUBLIC_CH}/messages",
        data=json.dumps({"content": TEXT, "allowed_mentions": {"parse": []}}).encode(),
        headers={"Authorization": f"Bot {token()}", "Content-Type": "application/json",
                 "User-Agent": "affine-notice/1.0"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        mid = json.loads(r.read())["id"]
    link = f"https://discord.com/channels/{GUILD}/{PUBLIC_CH}/{mid}"
    (Path(__file__).resolve().parent / "discord_wvk22_notice.links").open("a").write(f"public_reign15 {link}\n")
    print(link)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
