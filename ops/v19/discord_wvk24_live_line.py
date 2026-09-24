#!/usr/bin/env python3
"""Public + private "wvk 23 live" lines after the first wvk-23 verdict stamps.

  python ops/v17/discord_wvk22_live_line.py --flip-time "23:1x UTC" --first chal-00589 --margin 0.01 --se 0.028 --z 0.4 --seconds 3200
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
    assert len(text) <= 2000
    req = urllib.request.Request(
        f"https://discord.com/api/v10/channels/{ch}/messages",
        data=json.dumps({"content": text, "allowed_mentions": {"parse": []}}).encode(),
        headers={"Authorization": f"Bot {tok}", "Content-Type": "application/json",
                 "User-Agent": "affine-notice/1.0"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return f"https://discord.com/channels/{GUILD}/{ch}/{json.loads(r.read())['id']}"


def main() -> int:
    ap = argparse.ArgumentParser()
    for k in ("--flip-time", "--first", "--secs", "--forfeits", "--se", "--ctrl", "--box-commit", "--queue-n"):
        ap.add_argument(k, required=True)
    a = ap.parse_args()
    public = (f"**wvk 24 is live ({a.flip_time}).** Explicit operator directive (2026-09-23 20:17 UTC): the sd-meter **forfeit floor moves from −12 to −6 sd**. "
              f"A turn with no parseable action (or a thought with < 10 content tokens) now scores −6. Why: at −12 the ~2 % of turns that forfeit carried 48 % of the "
              f"per-turn score variance, dominating verdict SE and any training signal. −6 is still strictly worse than honest play — the 1st percentile of valid turns is "
              f"−4.6 sd, only 0.3 % of valid turns score below −6, and forfeiting even those perfectly would gain 0.007 sd/turn (3.5 % of δ) — so skipping a turn never pays. "
              f"Last 30 verdicts replayed: no decision changes, SE × 0.89. Nothing else changes (δ 0.2, k 2, 1,000 turns, caps 4,096/4,864, rendering, typicality prefix). "
              f"Two more items in the same fork, live from the next boundary (`chal-00679` on; `chal-00678` ran with the floor only): a teacher reference with fewer than 10 content tokens no longer anchors the typicality leg, and a turn with fewer than 2 such references scores min(z_R, z_A); and every verdict now publishes a k-matched, floor-dropped teacher-vs-king control (`control_kmatched`, overall + per leg) next to the legacy one. "
              f"Forward-only; **reign 21 stands**. First full-bundle verdict `{a.first}`: {a.secs} s, forfeits {a.forfeits}, SE {a.se}, control {a.ctrl}. "
              f"Details: https://affine.io/llms.txt → \"Fork history: wvk 24\".")
    private = (f"wvk 24 live {a.flip_time}: forfeit_sd −6 (was −12); box `{a.box_commit}`. Counterfactual ops/v19/floor_counterfactual.md: 0 flips / 30, SE ×0.89 median, genuine valid p1 −4.64, "
               f"0.32 % of valid turns below −6 (oracle gain 0.007 sd/turn). First verdict `{a.first}`: {a.secs} s, forfeits {a.forfeits}, SE {a.se}, control {a.ctrl}. Rollback on a control sign flip: bash ops/v19/rollback_wvk24.sh.")
    tok = token()
    links = [("public_live24", post(tok, PUBLIC_CH, public)), ("private_live24", post(tok, PRIVATE_CH, private))]
    (Path(__file__).resolve().parent / "discord_wvk24.links").open("a").write("".join(f"{k} {v}\n" for k, v in links))
    for k, v in links:
        print(k, v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
