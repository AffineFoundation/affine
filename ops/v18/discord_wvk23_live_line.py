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
    for k in ("--flip-time", "--first", "--secs", "--forfeits", "--lenz", "--ctrl", "--box-commit", "--queue-n"):
        ap.add_argument(k, required=True)
    a = ap.parse_args()
    public = (f"**wvk 23 is live ({a.flip_time}).** Explicit operator directive (2026-09-22 17:00 UTC), one fork, two changes: "
              f"(1) **`max_thought_tokens` 2,048 → 4,096** for your rollouts (teacher references 4,096 → 4,864 so they can think the full cap and act; "
              f"the teacher-relative cap `max(4096, 1.25×L_T)` stays). Kings have been thinking less every generation (GPQA chains 5.4k → 3.3k → 2.2k tokens "
              f"over reigns 19 → 21 while GPQA fell 83 → 74); the cap was not binding on D, so this alone will not reverse it — it removes the budget as a reason. "
              f"Duels take longer (~3× accepted). (2) **Typicality is one-sided on the long end:** the typicality leg reads only the first K content tokens of your "
              f"thought, K = the teacher's longest reference in content tokens. Extra deliberation beyond that is neither penalised nor paid; filler and off-task "
              f"reasoning still sit below the references, a pasted reference thought still sits above (both still lose). Offline probe on the last 30 verdicts: "
              f"no decision changes, 28 % of thoughts touched by +0.03–0.07 sd. Everything else unchanged (δ 0.2 sd, forfeit −12 sd, 1,000 turns, as-generated rendering, gates). "
              f"Forward-only; **reign 21 stands**; {a.queue_n} challenger(s) submitted under the old cap were judged under wvk 22 first. "
              f"First wvk-23 verdict `{a.first}`: {a.secs} s, forfeits {a.forfeits}, median thought chars {a.lenz}, control {a.ctrl}. "
              f"Full text + knobs: https://affine.io/llms.txt → \"Fork history: wvk 23\".")
    private = (f"wvk 23 live {a.flip_time}: max_thought_tokens 4096, ref_max_tokens 4864, sd_meter.content_prefix refs_max; box `{a.box_commit}`. "
               f"Probe: ops/v18/probe_a.txt (30 verdicts, (i) one-sided: 0 decision changes; overall teacher-vs-king control already NEGATIVE on all 30, z −3.4…−8.6 — kings beat the teacher's held-out replies on R/A; typ leg still +0.36 for the teacher), "
               f"probe_b.txt (173-turn re-echo, (ii) refs_max: bites 28 % of thoughts, Δtyp +0.03…+0.07, typ-leg control +0.36 → +0.27 sd, z 3.8 → 3.0). "
               f"First verdict `{a.first}`: {a.secs} s, forfeits {a.forfeits}, len_z {a.lenz}, control {a.ctrl}. Rollback on a control SIGN FLIP vs pre-fork: bash ops/v18/rollback_wvk23.sh.")
    tok = token()
    links = [("public_live23", post(tok, PUBLIC_CH, public)), ("private_live23", post(tok, PRIVATE_CH, private))]
    (Path(__file__).resolve().parent / "discord_wvk23.links").open("a").write("".join(f"{k} {v}\n" for k, v in links))
    for k, v in links:
        print(k, v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
