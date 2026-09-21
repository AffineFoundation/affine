#!/usr/bin/env python3
"""Public + private lines for the 19:40 UTC directive: rendering folded in, flip at the next boundary."""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

GUILD = "799672011265015819"
PUBLIC_CH = "1381987595881414656"
PRIVATE_CH = "1510910974498967613"
REPO = Path(__file__).resolve().parents[2]
PUBLIC = ("**wvk 22 — two changes (operator directive 19:40 UTC).** (1) **Effective at the NEXT duel boundary** "
          "(≈ 20:45 UTC), not after the queue: the duel in flight (`chal-00586`) finishes under wvk 21; **everything "
          "queued from `chal-00587` on is judged under wvk 22.** Reign 15 stands; forward-only. (2) **Thoughts are scored "
          "as generated.** Until now every echo rendered your reasoning as prose after `</think>` (`</think>\\nTHOUGHT: …`). "
          "From wvk 22 the reply is rendered the way you produced it — `<think>{reasoning}\\n</think>\\n\\n{visible thought}\\n\\n{action}` — "
          "for the teacher references and both sides alike, reasoning and visible spans scored, visible text verbatim (no label added or "
          "stripped). Why: the old body scored the teacher's own visible sentence at −0.18 nats/byte (as generated: −0.06), so the meter "
          "could not tell the teacher's held-out reply from a reasoning-only king (control z 1.2 → 8.7). **What you must do:** reason "
          "inside `<think>…</think>`, then write a short visible thought, then the action. Reasoning-only replies are atypical on most "
          "turns under the new typicality leg; pasting the reasoning again after `</think>` scores lower, not higher (checked before the "
          "flip). Final knobs stamped in llms.txt → \"Fork history: wvk 22\" at the flip.")
PRIVATE = ("wvk 22 flip armed for the next boundary (~20:45 UTC, after chal-00586; SKIP_CUTOFF_WAIT=1, box `ac369a9`): "
           "score_mode sd_min_rga, n_turns 1000, thought_rendering as_generated, δ_sd 0.20, k_sigma 2, forfeit −12 sd. "
           "Calibration under (b) from the 225-turn re-echo + 3 (a) shadow duels: kings' typ_c mean −1.1…−1.3, sd 2.9, p1 −12, 59 % "
           "of turns outside 2σ; paired chal−king sd_diff ≈ 2.4–3.1 → SE@1000 ≈ 0.08–0.10, δ = 0.082·sd_diff ≈ 0.20–0.25; floor under "
           "p1 and '2 % forfeit = one δ' both ≈ −12. Predicted: teacher-vs-king ≈ +2 sd (z ≫ 10); a latent-only challenger vs the king "
           "≈ −0.06 ± 0.08 (no crown); a latent+visible teacher-like challenger ≈ +2 sd → crowns. Pad-after-</think> arm with the NEW "
           "code on the live swarm (75 turns): repeat −1.29 ± 0.36, tail −1.24, generic −2.14 typ_c vs honest → the lever does not pay. "
           "Frozen anchors are (a)-calibrated → not headlined (anchor = loo). Rollback: bash ops/v17/rollback_wvk22.sh (reverts rendering too).")


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
    tok = token()
    links = [("public_now", post(tok, PUBLIC_CH, PUBLIC)), ("private_now", post(tok, PRIVATE_CH, PRIVATE))]
    (Path(__file__).resolve().parent / "discord_wvk22_notice.links").open("a").write("".join(f"{k} {v}\n" for k, v in links))
    for k, v in links:
        print(k, v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
