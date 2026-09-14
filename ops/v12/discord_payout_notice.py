#!/usr/bin/env python3
"""Post the ONE public payout-rule notice to the SN120 Discord channel.

    python ops/v12/discord_payout_notice.py            # preview (no post)
    python ops/v12/discord_payout_notice.py --post     # post once

Reads ops/v12/discord_post.md, fills {EFFECTIVE} from
[subnet].king_payout_rule_effective_at and {PAID_SET} from the live
snapshot, posts with the Arbos bot token (DISCORD_BOT_TOKEN_ARBOS_BITTENSOR
from the environment or the repo .env) and prints the message link.
Public channel per the wvk-16 / corpus announce tooling (operator
instruction 2026-09-14 11:09 UTC: "post ONE public notice").
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tomllib
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
POST_MD = Path(__file__).with_name("discord_post.md")
GUILD_ID = "799672011265015819"
CHANNEL_ID = "1381987595881414656"  # public "120・ⴷffine・ⴷ" channel
SNAPSHOT_URL = "http://127.0.0.1:8787/api/v1/snapshot"


def env_value(name: str) -> str:
    if os.environ.get(name):
        return os.environ[name]
    env = REPO / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip().strip('"')
    return ""


def effective_at() -> str:
    with open(REPO / "affine" / "affine.toml", "rb") as f:
        sub = tomllib.load(f)["subnet"]
    eff = str(sub.get("king_payout_rule_effective_at") or "").strip()
    return eff.replace("+00:00", "Z").replace("T", " ").rstrip("Z")[:16] if eff else "(unset)"


def paid_set() -> str:
    try:
        with urllib.request.urlopen(SNAPSHOT_URL, timeout=15) as r:
            snap = json.load(r)
    except Exception as e:  # noqa: BLE001
        return f"(snapshot unavailable: {e})"
    p = snap.get("payout") or {}
    if p.get("burn") or not p.get("paid"):
        return "no crown inside its window → burn"
    parts = []
    for m in p["paid"]:
        uid = f"uid {m['uid']}" if m.get("uid") is not None else "unregistered"
        parts.append(f"reign {m.get('reign_number')} ({uid}) {round(100 * float(m.get('share') or 0))} %")
    return "; ".join(parts)


def build() -> str:
    text = POST_MD.read_text(encoding="utf-8")
    return text.replace("{EFFECTIVE}", effective_at()).replace("{PAID_SET}", paid_set())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--post", action="store_true", help="actually post")
    args = ap.parse_args()
    content = build()
    if len(content) > 2000:
        print(f"message is {len(content)} chars (> 2000 Discord limit)", file=sys.stderr)
        sys.exit(1)
    print(content)
    print(f"\n[{len(content)} chars]", file=sys.stderr)
    if not args.post:
        return
    token = env_value("DISCORD_BOT_TOKEN_ARBOS_BITTENSOR")
    if not token:
        print("DISCORD_BOT_TOKEN_ARBOS_BITTENSOR not found", file=sys.stderr)
        sys.exit(1)
    req = urllib.request.Request(
        f"https://discord.com/api/v10/channels/{CHANNEL_ID}/messages",
        data=json.dumps({"content": content}).encode(),
        headers={"Authorization": f"Bot {token}", "Content-Type": "application/json",
                 "User-Agent": "affine-ops (https://affine.io, 1.0)"},
        method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        mid = json.load(r).get("id")
    link = f"https://discord.com/channels/{GUILD_ID}/{CHANNEL_ID}/{mid}"
    print(f"posted: {link}", file=sys.stderr)
    Path(__file__).with_name("discord_post.link").write_text(link + "\n")


if __name__ == "__main__":
    main()
