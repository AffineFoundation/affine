#!/usr/bin/env python3
"""Public + private Discord lines for the operator crown of chal-00687 (reign 22).

  python ops/v20/discord_operator_crown_00687.py --crowned-at "19:2x UTC" --weights-at "19:3x UTC" --public-url URL
Links → ops/v20/discord_operator_crown_00687.links
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path

GUILD = "799672011265015819"
PUBLIC_CH = "1381987595881414656"
PRIVATE_CH = "1510910974498967613"
REPO = Path(__file__).resolve().parents[2]


def token() -> str:
    t = os.environ.get("DISCORD_BOT_TOKEN_ARBOS_BITTENSOR")
    if t:
        return t
    for line in (REPO / ".env").read_text().splitlines():
        if line.startswith("DISCORD_BOT_TOKEN_ARBOS_BITTENSOR="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("no token")


def post(tok: str, ch: str, text: str) -> str:
    assert len(text) <= 2000, len(text)
    req = urllib.request.Request(
        f"https://discord.com/api/v10/channels/{ch}/messages",
        data=json.dumps({"content": text, "allowed_mentions": {"parse": []}}).encode(),
        headers={"Authorization": f"Bot {tok}", "Content-Type": "application/json",
                 "User-Agent": "affine-notice/1.0"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return f"https://discord.com/channels/{GUILD}/{ch}/{json.loads(r.read())['id']}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crowned-at", required=True)
    ap.add_argument("--weights-at", required=True)
    ap.add_argument("--public-url", required=True)
    ap.add_argument("--private-only", action="store_true")
    a = ap.parse_args()
    public = (
        f"**Reign 22 — operator crown ({a.crowned_at}).** Explicit operator directive (Jacob Steeves, 2026-09-26 18:51 UTC): "
        f"\"crown the last miner model who scored the best against the king and set weights to it immediately while we consider the cut over.\"\n\n"
        f"**Crowned:** `chal-00687` — uid 62, hotkey `5EzaX8pVDyqC…`, digest `7f066f2c5f95…` — the best challenger against reign 21 since its crown "
        f"(2026-09-22): paired margin **+0.073 sd**, SE 0.027, **z +2.73** (duelled 2026-09-24 19:46 UTC). It cleared the 2·SE statistical bar but not "
        f"the δ = 0.20 sd floor, so under the wvk-24 rule it was not crowned; every other challenger since then scored lower (runner-ups +0.054 z 1.2, +0.047 z 1.3). "
        f"Probe 0.90 pass, forfeits 0.3 %, arch pin + hygiene passed; not a byte/ε-copy of any king (0/18 shard hashes shared; tensor sample vs reign 21 differs "
        f"densely, ≈1e-3 relative update). Reign 21's 72 h payout window had expired on 09-25, so weights had been burning.\n\n"
        f"**What changed:** the king in `state.json`, one `crowned` row with `via = \"operator_crown\"` and the note *\"operator crown 2026-09-26; did not clear δ under wvk 24\"*, "
        f"weights set to uid 62 at {a.weights_at}, model public at {a.public_url}. **Nothing else:** no scoring change, `weight_version_key` stays 24, every duel from now "
        f"runs against reign 22 under the same rule; the wvk-25 fork (GLM teacher, 262k, 2026-09-30 14:00 UTC) is unchanged. Details: https://affine.io/llms.txt → \"Operator crown 2026-09-26\"."
    )
    private = (
        f"[ops] operator crown done ({a.crowned_at}): chal-00687 → reign 22 (uid 62, 7f066f2c5f95). Path = ops/v20/operator_crown_00687.{{py,sh}} "
        f"(reign-14 retro path: promote → record_verdict via=operator_crown → bench card). Validator stopped at an empty boundary (queue 0, no in_flight), deadman paused, "
        f"keepalive off/on, no pod redeploy. Weights: {a.weights_at}. King seat: kingctl follows state.json (watch for king-dg-7f066f2c5f95 box). "
        f"Copy check: file hashes vs reigns 14–21 0/18 shared; tensor sample vs 21: median 30 % elements changed, ‖Δ‖/‖king‖ ≈ 1e-3; coldkey differs from the grpo lineage. "
        f"No wvk change; wvk-25 plan untouched (reign 22 stands at T0 instead of 21 — lead: update the plan/notice wording)."
    )
    tok = token()
    links = []
    if not a.private_only:
        links.append(post(tok, PUBLIC_CH, public))
    links.append(post(tok, PRIVATE_CH, private))
    (REPO / "ops/v20/discord_operator_crown_00687.links").write_text("\n".join(links) + "\n")
    print("\n".join(links))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
