#!/usr/bin/env python3
"""Public + private "wvk 22 live" lines after the first wvk-22 verdict stamps.

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
    for k in ("--flip-time", "--first", "--margin", "--se", "--z", "--seconds", "--box-commit"):
        ap.add_argument(k, required=True)
    a = ap.parse_args()
    public = (f"**wvk 22 is live ({a.flip_time}).** The scoring rule is now the sd-meter "
              f"`min(z_R, typ_c, z_A)` in teacher-sd units on **1,000-turn** slices, with thoughts **scored as generated** "
              f"(`<think>reasoning</think>` + visible thought + action, for the teacher references and both sides): δ = **0.20 sd**, "
              f"k_sigma = 2, forfeit floor = **−12 sd** (calibrated on the as-generated re-echo of 225 stored turns: δ = 0.082 × the "
              f"per-turn diff sd; floor under the 1st percentile of the kings' valid turns). "
              f"Duels through `chal-00586` were judged under wvk 21; everything from `chal-00587` on under wvk 22. **Reign 15 stands**; "
              f"forward-only, no re-verdicts. First wvk-22 verdict `{a.first}`: margin {a.margin} sd, SE {a.se}, "
              f"z {a.z}, {a.seconds} s. Every verdict's `margin / se / z` are now in sd units; the leg breakdown "
              f"(z_R, typ_c, z_A, bind fractions, teacher control) is under `shadow.sd_meter` with `role = \"rule\"`. "
              f"Full definition + knobs: https://affine.io/llms.txt → \"Fork history: wvk 22\".")
    private = (f"wvk 22 live {a.flip_time}: score_mode sd_min_rga, thought_rendering as_generated, n_turns 1000, δ_sd 0.20, k_sigma 2, forfeit −12 sd; "
               f"box `{a.box_commit}`, PR #36. Duels through chal-00586 judged under wvk 21; reign 15 stands. "
               f"First verdict `{a.first}`: margin {a.margin} sd / SE {a.se} / z {a.z} / {a.seconds} s. Rollback rule "
               f"on the first 3 verdicts (SE > 2× shadow, teacher-vs-king z ≤ −2, leg binds > 80 %, leg dropped > 5 %): "
               f"`bash ops/v17/rollback_wvk22.sh`.")
    tok = token()
    links = [("public_live", post(tok, PUBLIC_CH, public)), ("private_live", post(tok, PRIVATE_CH, private))]
    (Path(__file__).resolve().parent / "discord_wvk22_notice.links").open("a").write(
        "".join(f"{k} {v}\n" for k, v in links))
    for k, v in links:
        print(k, v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
