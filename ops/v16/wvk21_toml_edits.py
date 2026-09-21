"""wvk 20 -> 21: remove the confirmation slice — crown on one slice again.

Explicit dated operator directive (Jacob Steeves, 2026-09-17 10:07 UTC):
"Remove the double eval on kings. This is too difficult. Lets crown if any
model passes 2 sigma like before. Make the change and in llms.txt the update
makes things too hard. Feel free to crown the last model which passed but
failed the crown."

What flips:
  confirmation_required   true -> false   a duel that clears
                                          max(k_sigma·SE, min_margin) + gates
                                          crowns at once (the wvk 3–18 rule)
  weight_version_key      20 -> 21        (+ one history paragraph)
Asserted, not edited: k_sigma = 2.0, min_margin = 0.002, near_miss_enabled =
false, crown_mode = "duel", thought_cap_ratio = 1.25, max_thought_tokens =
2048, ref_max_tokens = 4096.

    python ops/v16/wvk21_toml_edits.py --preview
    python ops/v16/wvk21_toml_edits.py --apply 2026-09-17 --wvk-to 21
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
PATCH = REPO / "ops" / "v16" / "wvk21_no_confirmation.patch"

OLD = "confirmation_required = true\n"
NEW = (
    "# {date} (weight_version_key {a}→{b}, explicit dated operator directive\n"
    "# 2026-09-17 10:07 UTC: \"Remove the double eval on kings. This is too\n"
    "# difficult. Lets crown if any model passes 2 sigma like before\"): OFF.\n"
    "# One 1,300-turn slice decides again — margin > max(k_sigma·SE, δ) +\n"
    "# gates crowns at once. The one confirmation that ran live (chal-00556,\n"
    "# 2026-09-16: slice 1 +0.0022 z 3.13, slice 2 +0.0007 z 1.03, pooled\n"
    "# +0.0014 < δ) is reversed by the same directive: chal-00556 crowned\n"
    "# retroactively from its stored slice-1 verdict. Code kept for replay.\n"
    "confirmation_required = false\n")

ASSERT_LINES = (
    "k_sigma = 2.0\n",
    "min_margin = 0.002\n",
    "near_miss_enabled = false\n",
    'crown_mode = "duel"\n',
    "thought_cap_ratio = 1.25\n",
    "max_thought_tokens = 2048\n",
    "ref_max_tokens = 4096\n",
)

BANNER = "# ############################################################################\n"
HISTORY = (
    "# {a}→{b} confirmation slice removed ({date}): explicit dated operator\n"
    "# directive 2026-09-17 10:07 UTC — \"Remove the double eval on kings. This\n"
    "# is too difficult. Lets crown if any model passes 2 sigma like before.\n"
    "# Make the change and in llms.txt the update makes things too hard. Feel\n"
    "# free to crown the last model which passed but failed the crown.\"\n"
    "# confirmation_required → false: a challenger crowns iff its paired\n"
    "# margin over ONE 1,300-turn slice clears max(k_sigma·SE, min_margin) =\n"
    "# max(2·SE, 0.002) plus the thought-length floor and B gate — the rule of\n"
    "# wvk 3–18. The teacher-relative thought cap (wvk 20), caps, min(R,G)\n"
    "# math and the reign chain are unchanged; near_miss stays off. Retroactive\n"
    "# crown: chal-00556 (uid 175, 0f4029fd…), whose slice 1 cleared the bar\n"
    "# (+0.0022, z 3.13) and whose confirmation slice fell short (pooled\n"
    "# +0.0014 < δ), is crowned from its stored slice-1 verdict — the only\n"
    "# confirmation-only rejection since wvk 19. Forward-only otherwise;\n"
    "# min_submission_block unchanged.\n")


def render(src: str, date: str, wvk_to: int) -> str:
    a, b = wvk_to - 1, wvk_to
    fmt = dict(date=date, a=a, b=b)
    for line in ASSERT_LINES:
        if src.count(line) != 1:
            raise SystemExit(f"anchor changed, refusing: expected exactly one {line.strip()!r}")
    if src.count(OLD) != 1:
        raise SystemExit("anchor missing or already flipped: confirmation_required = true")
    key_old = f"weight_version_key = {a}\n"
    if src.count(key_old) != 1:
        raise SystemExit(f"weight_version_key is not {a}")
    out = src.replace(OLD, NEW.format(**fmt))
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
    g.add_argument("--apply", metavar="DATE")
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
    print(f"applied wvk {args.wvk_to - 1} -> {args.wvk_to}: confirmation_required = false "
          f"({date}){mirrored}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
