"""wvk 17 -> 18: miner thought cap 1024 -> 2048 + prose `text` fallback at tool_call turns.

Operator directive (Jacob Steeves, 2026-09-15 20:12 UTC, "yes lets go").

What flips:
  max_thought_tokens            1024 -> 2048   miners may think up to 2,048 tokens
                                               (max_action_tokens 768 unchanged;
                                               teacher ref_max_tokens 4096 unchanged)
  text_fallback_at_tool_turns   (absent = false) -> true
                                               at a tool_call turn a closed-think reply
                                               with no tool call but a non-empty visible
                                               reply is a `text` action — teacher reference
                                               and miner alike (no drop, no forfeit)
  weight_version_key            17 -> 18       (+ one history paragraph)
Asserted, not edited: max_action_tokens = 768, ref_max_tokens = 4096,
require_think_close = true, band_c = 4.0, k_sigma = 2.0, min_margin = 0.002,
crown_mode = "duel".

    python ops/v12/wvk18_toml_edits.py --preview                 # ops/v12/wvk18_cap_textfallback.patch
    python ops/v12/wvk18_toml_edits.py --apply 2026-09-15 --wvk-to 18

The website mirror `affine/website/code/affine.toml` is overwritten with the
result when it exists. --apply refuses unless every anchor is in its
pre-flip state and weight_version_key == N-1.
"""

from __future__ import annotations

import argparse
import difflib
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TOML = REPO / "affine" / "affine.toml"
MIRROR = REPO / "affine" / "website" / "code" / "affine.toml"
PATCH = REPO / "ops" / "v12" / "wvk18_cap_textfallback.patch"

CAP_OLD = "max_thought_tokens = 1024\nmax_action_tokens = 768\n"
CAP_NEW = (
    "# {date} (weight_version_key {a}→{b}, explicit dated operator directive\n"
    "# 2026-09-15 20:12 UTC): miner thought cap 1024 → 2048. On fresh samples\n"
    "# the plain teacher's own forfeits at 1,024 fall 21% → 7% at 2,048 and a\n"
    "# coached teacher's 29% → 16% (internal/hints/coach2/cap_arm.md,\n"
    "# docs/frontier-coach-design.md §4.2a): a miner that thinks past 1,024\n"
    "# tokens forfeited for nothing the meter cares about. Action cap and the\n"
    "# teacher's ref_max_tokens are unchanged; the serving window holds\n"
    "# (110k-token prefix cap + 2048 + 768 < 131072).\n"
    "max_thought_tokens = 2048\n"
    "max_action_tokens = 768\n")
TF_OLD = "require_think_close = true\n"
TF_NEW = (
    "require_think_close = true\n"
    "# {date} (weight_version_key {a}→{b}): at a tool_call turn a reply that\n"
    "# CLOSED </think>, calls no tool but says something visible is a `text`\n"
    "# action (the whole visible reply) instead of a dropped reference / a\n"
    "# miner forfeit — the rule the fold's teacher probe already applies,\n"
    "# now at duel time for every turn, teacher and miner alike. The\n"
    "# teacher's own samples at tool_call turns are prose 14% of the time\n"
    "# (58% of its non-tool samples, teacher_probe, 86k samples); over the\n"
    "# last 20 verdicts ~69 reference slots per verdict were empty at\n"
    "# tool_call turns and 17 / 23 miner turns per side forfeited there. An\n"
    "# empty visible reply is still a forfeit; an unclosed think block never\n"
    "# reaches the fallback (the wvk-13 reasoning-only hole stays closed).\n"
    "# false = pre-wvk-18 replay path.\n"
    "text_fallback_at_tool_turns = true\n")

ASSERT_LINES = (
    "k_sigma = 2.0\n",
    "min_margin = 0.002\n",
    'crown_mode = "duel"\n',
    "ref_max_tokens = 4096\n",
    "band_c = 4.0\n",
)

BANNER = "# ############################################################################\n"
HISTORY = (
    "# {a}→{b} miner thought cap + prose fallback at tool turns ({date}): explicit\n"
    "# dated operator directive 2026-09-15 20:12 UTC (\"yes lets go\"). (1)\n"
    "# max_thought_tokens 1024 → 2048 (action cap 768 and the teacher's\n"
    "# ref_max_tokens 4096 unchanged): on fresh samples the plain teacher's own\n"
    "# forfeits at 1,024 tokens fall 21% → 7% at 2,048 and a coached teacher's\n"
    "# 29% → 16% — thinking past 1,024 tokens forfeited for nothing the meter\n"
    "# cares about. (2) text_fallback_at_tool_turns = true: at a tool_call turn\n"
    "# a reply that closed </think>, calls no tool but says something visible\n"
    "# is a `text` action (whole visible reply) for teacher references and\n"
    "# miners alike, instead of a dropped reference / a forfeit; the teacher\n"
    "# itself answers in prose on 14% of its samples at such turns. Empty\n"
    "# visible reply = still a forfeit; unclosed </think> never reaches the\n"
    "# fallback. min(R,G) math, the crown bar max(2·SE, 0.002), gates and the\n"
    "# reign chain are unchanged; wvk ≤ 17 verdicts replay through their\n"
    "# stamped caps and the knob's false default. Forward-only; reign 13\n"
    "# stands; min_submission_block unchanged.\n")


def render(src: str, date: str, wvk_to: int) -> str:
    a, b = wvk_to - 1, wvk_to
    fmt = dict(date=date, a=a, b=b)
    for line in ASSERT_LINES:
        if src.count(line) != 1:
            raise SystemExit(f"anchor changed, refusing: expected exactly one {line.strip()!r}")
    for anchor in (CAP_OLD, TF_OLD):
        if src.count(anchor) != 1:
            raise SystemExit(f"anchor missing or already flipped: {anchor.splitlines()[0]!r}")
    if "text_fallback_at_tool_turns" in src:
        raise SystemExit("text_fallback_at_tool_turns already present in the toml")
    key_old = f"weight_version_key = {a}\n"
    if src.count(key_old) != 1:
        raise SystemExit(f"weight_version_key is not {a}")
    out = src.replace(CAP_OLD, CAP_NEW.format(**fmt))
    out = out.replace(TF_OLD, TF_NEW.format(**fmt))
    out = out.replace(key_old, f"weight_version_key = {b}\n")
    idx = out.find(BANNER)
    if idx < 0 or out.find(BANNER, idx + 1) < 0:
        raise SystemExit("anchor missing: weight_version_key banner")
    head, tail = out[:idx], out[idx:]
    if not head.endswith("#\n"):
        raise SystemExit("unexpected text above the banner")
    return head[:-2] + HISTORY.format(**fmt) + "#\n" + tail


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--toml", type=Path, default=TOML)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--preview", action="store_true")
    g.add_argument("--apply", metavar="DATE", help="explicit dated operator directive (YYYY-MM-DD)")
    ap.add_argument("--wvk-to", type=int)
    ap.add_argument("--no-mirror", action="store_true")
    args = ap.parse_args()
    src = args.toml.read_text()
    if args.wvk_to is None:
        m = re.search(r"^weight_version_key = (\d+)$", src, re.M)
        args.wvk_to = int(m.group(1)) + 1 if m else 0
    date = args.apply or "YYYY-MM-DD"
    if args.apply and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.apply):
        raise SystemExit("--apply needs a YYYY-MM-DD directive date")
    out = render(src, date, args.wvk_to)
    if args.preview:
        PATCH.write_text("".join(difflib.unified_diff(
            src.splitlines(True), out.splitlines(True),
            "a/affine/affine.toml", "b/affine/affine.toml")))
        print(f"wrote {PATCH}")
        return 0
    args.toml.write_text(out)
    mirrored = ""
    if not args.no_mirror and MIRROR.exists() and args.toml.resolve() == TOML.resolve():
        shutil.copyfile(args.toml, MIRROR)
        mirrored = f"; mirrored to {MIRROR.relative_to(REPO)}"
    print(f"applied wvk {args.wvk_to - 1} -> {args.wvk_to}: max_thought_tokens = 2048, "
          f"text_fallback_at_tool_turns = true ({date}){mirrored}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
