#!/usr/bin/env python3
"""Public + private Discord lines for the wvk 25 δ fork (δ 0.20 → 0.10 sd), posted after the flip.

  python ops/v21/discord_wvk25_delta.py --flip-time "HH:MM UTC" [--first chal-xxxxx] [--private-only]
Links → ops/v21/discord_wvk25_delta.links
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
    ap.add_argument("--flip-time", required=True)
    ap.add_argument("--first", default="")
    ap.add_argument("--private-only", action="store_true")
    a = ap.parse_args()
    first = f" First wvk-25 duel: `{a.first}`." if a.first else ""
    public = (
        f"**wvk 25 is live ({a.flip_time}): crown floor δ 0.20 → 0.10 sd.** Explicit operator directive (Jacob Steeves, 2026-09-27 08:22 UTC: "
        f"\"Lower the validator margin to 0.1\"). A challenger now crowns when its paired margin over the 1,000-turn slice clears **max(2·SE, 0.10 sd)** "
        f"and the gates pass (thought floor, B licence, protocol probe). One number changes; scores, legs, anchors, the −6 forfeit floor, the caps and the "
        f"rendering are untouched — a stored verdict replays to the same margin/SE/z, only the crown decision moves. **Reign 22 stands**; forward-only, "
        f"no re-verdicts, `min_submission_block` unchanged.{first}\n\n"
        f"**What it would have done** (108 verdicts since wvk 22, each vs its then-king): 8 more crowns — chal-00613 +0.197 (z 2.6), 00631 +0.154 (3.3), "
        f"00643 +0.157 (3.6), 00649 +0.135 (3.2), 00651 +0.166 (3.5), 00652 +0.174 (4.1), 00653 +0.186 (3.6), 00655 +0.112 (2.5); the 7 real crowns stand; "
        f"chal-00687 (+0.073, reign 22 by operator crown) stays under.\n\n"
        f"**Known risk, chosen by the operator:** at the live noise (SE median 0.047 sd at n≈1000) δ = 0.10 is ≈ 2.1 SE (0.20 was ≈ 4.2), so for the noisier "
        f"half of duels the 2σ test alone decides — the same shape as the 2026-08-21/22 δ experiment that produced near-noise crowns and winner's-curse churn "
        f"and was reverted the next day. If churn returns, the revert is its own fork.\n\n"
        f"**Numbering:** the GLM-5.3-Flash / 262k / scoring-bundle fork announced yesterday as \"wvk 25\" keeps its date (**Wed 2026-09-30 14:00 UTC**) and "
        f"content and becomes **wvk 26**. Spec: https://affine.io/llms.txt → \"Fork history: wvk 25\" and \"Upcoming fork: wvk 26\"."
    )
    private = (
        f"[ops] wvk 24→25 flipped {a.flip_time}: [duel.sd_meter].min_margin_sd 0.2 → 0.1 (Jacob 08:22 UTC). Path ops/v21/deploy_wvk25_delta.sh "
        f"(env from /proc, keepalive off, deadman paused, pm2 stop at an empty boundary, toml + llms + banner, redeploy_pods --role eval, pm2 start; "
        f"king before/after = chal-00687 reign 22). Rollback = ops/v21/rollback_wvk25_delta.sh (operator call on churn: near-noise crowns / winners' "
        f"score drifting down). Watch: crowns/day, z of crowns, control_matched. **Lead (bc-3979ad9e): Wednesday's bundle is wvk 26** — bump 25→26 in "
        f"teacher_swap_toml_edits / wvk25_toml_edits / deploy_teacher_swap.sh, assert min_margin_sd 0.1, notice text \"reign 22 stands\"; my "
        f"ops/v20/wvk25_rules_toml_edits.py already writes the wvk-26 paragraph and asserts δ 0.1. New artifact field turn_kinds (per-turn action_kind) "
        f"for exact offline replay (heuristic kinds drifted margins ≤ 0.02 sd).{first}"
    )
    tok = token()
    links = []
    if not a.private_only:
        links.append(post(tok, PUBLIC_CH, public))
    links.append(post(tok, PRIVATE_CH, private))
    (REPO / "ops/v21/discord_wvk25_delta.links").write_text("\n".join(links) + "\n")
    print("\n".join(links))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
