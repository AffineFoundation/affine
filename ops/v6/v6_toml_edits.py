"""Stage the v6 scoring flip (min(R,G,A) + forfeit floor) in affine.toml.

Built 2026-09-04 on an operator directive to implement both rules; the
FLIP itself is a weight_version_key event and only runs with an explicit,
dated directive that names the key (AGENTS.md / .cursor/rules). This script
renders the flipped contract from whatever tree it is given, so it works
before or after the wvk-11 T0 edit (ops/t0/t0_toml_edits.py).

    python ops/v6/v6_toml_edits.py --preview [--toml PATH]
    python ops/v6/v6_toml_edits.py --apply DATE --wvk-to N [--toml PATH]

--apply refuses unless score_mode is "min_rg", no forfeit_turn_score line
exists, and weight_version_key == N-1.

What flips (see the [duel] comment block in affine.toml for the why):
  score_mode           "min_rg" -> "min_rga"   (turn = min(R, G, A))
  forfeit_turn_score   absent   -> -0.1        (no parseable action = floor)
  weight_version_key   N-1      -> N           (+ one history line)
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TOML = REPO / "affine" / "affine.toml"

MODE_OLD = 'score_mode = "min_rg"\n'
MODE_NEW = (
    "# {date} (weight_version_key {a}→{b}, explicit dated operator directive):\n"
    "# min(R,G,A) — the action leg joins the min; see the v6 block below.\n"
    'score_mode = "min_rga"\n')

FLOOR_ANCHOR = "band_floor = 0.002\n"
FLOOR_NEW = (
    "band_floor = 0.002\n"
    "# {date} (weight_version_key {a}→{b}): forfeits are scored, not dropped.\n"
    "forfeit_turn_score = -0.1\n")

BANNER = "# ############################################################################\n"
HISTORY_LINE = (
    "# {a}→{b} min(R,G,A) v6 + forfeit floor ({date}): the miner's action is\n"
    "# ranked (A = tempered LME of lpC(y_A|z_C^i) − lpC(y_A|∅)), and a turn with\n"
    "# no parseable action scores forfeit_turn_score instead of leaving the\n"
    "# pairing. Forward-only; reign stands.\n")


HISTORY_LINE_FORFEIT_ONLY = (
    "# {a}→{b} forfeit floor ({date}): a turn with no parseable action scores\n"
    "# forfeit_turn_score (−0.1) instead of leaving the pairing; one-sided =\n"
    "# loss, two-sided = tie. min(R,G) unchanged. Forward-only; reign stands.\n")


def render(src: str, date: str, wvk_to: int, action_leg: bool = True) -> str:
    a, b = wvk_to - 1, wvk_to
    if action_leg and src.count(MODE_OLD) != 1:
        raise SystemExit('anchor missing: score_mode = "min_rg"')
    if re.search(r"^forfeit_turn_score\s*=", src, re.M):
        raise SystemExit("forfeit_turn_score already set")
    key_old = f"weight_version_key = {a}\n"
    if src.count(key_old) != 1:
        raise SystemExit(f"weight_version_key is not {a}")
    if src.count(FLOOR_ANCHOR) != 1:
        raise SystemExit("anchor missing: band_floor = 0.002")
    fmt = dict(date=date, a=a, b=b)
    out = src
    if action_leg:
        out = out.replace(MODE_OLD, MODE_NEW.format(**fmt))
    out = out.replace(FLOOR_ANCHOR, FLOOR_NEW.format(**fmt))
    out = out.replace(key_old, f"weight_version_key = {b}\n")
    # History line: right before the NEVER-CHANGE banner that precedes the key.
    idx = out.find(BANNER)
    if idx < 0 or out.find(BANNER, idx + 1) < 0:
        raise SystemExit("anchor missing: weight_version_key banner")
    # The history paragraph ends with a bare "#\n" line just above the banner.
    head, tail = out[:idx], out[idx:]
    if not head.endswith("#\n"):
        raise SystemExit("unexpected text above the banner")
    hist = HISTORY_LINE if action_leg else HISTORY_LINE_FORFEIT_ONLY
    out = head[:-2] + hist.format(**fmt) + "#\n" + tail
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--toml", type=Path, default=TOML)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--preview", action="store_true")
    g.add_argument("--apply", metavar="DATE",
                   help="explicit dated operator directive (YYYY-MM-DD)")
    ap.add_argument("--wvk-to", type=int, help="new weight_version_key")
    ap.add_argument("--forfeit-only", action="store_true",
                    help="flip only forfeit_turn_score; leave score_mode at "
                         "min_rg (the 2026-09-04 probe found the A leg "
                         "rewards short actions ~10x — see AGENTS.md)")
    args = ap.parse_args()
    src = args.toml.read_text()
    if args.wvk_to is None:
        m = re.search(r"^weight_version_key = (\d+)$", src, re.M)
        args.wvk_to = int(m.group(1)) + 1 if m else 0
    date = args.apply or "YYYY-MM-DD"
    if args.apply and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.apply):
        raise SystemExit("--apply needs a YYYY-MM-DD directive date")
    out = render(src, date, args.wvk_to, action_leg=not args.forfeit_only)
    if args.preview:
        sys.stdout.writelines(difflib.unified_diff(
            src.splitlines(True), out.splitlines(True),
            "a/affine/affine.toml", "b/affine/affine.toml"))
        return 0
    args.toml.write_text(out)
    what = ("forfeit_turn_score -0.1" if args.forfeit_only
            else "score_mode min_rga, forfeit_turn_score -0.1")
    print(f"applied v6 flip: weight_version_key {args.wvk_to - 1} -> {args.wvk_to}, "
          f"{what} ({date})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
