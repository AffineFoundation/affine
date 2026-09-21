"""wvk 15 -> 16: restore the per-duel crown rule in affine.toml.

Operator directive (Jacob Steeves, 2026-09-13 12:10 UTC): "go back to the
original delta threshold, the original system we had more than a day ago;
uncrown the copied model; announce in Discord once all of this has
happened." Trigger: reign 13 (chal-00461) was a byte-for-byte re-upload of
reign 12's weights that crowned under the 12 h window rule on a z = 0.93
margin and a pooled confirmation of +0.000045 (z 0.08).

What flips (the mechanism stays in code; pre-fork verdicts still replay):
  crown_mode           "window_best" -> "duel"   every duel crowns on its own
                                                 bar max(k_sigma·SE, δ) + gates
  near_miss_enabled    true -> false             one seeded slice decides
  weight_version_key   15 -> 16                  (+ one history paragraph)
Asserted, not edited: k_sigma = 2.0, min_margin = 0.002,
min_margin_mode = "fixed" (decaying δ off), min_z = 0.0.

    python ops/v10/wvk16_toml_edits.py --preview                 # ops/v10/wvk16_restore_duel.patch
    python ops/v10/wvk16_toml_edits.py --apply 2026-09-13 --wvk-to 16

The website mirror `affine/website/code/affine.toml` is overwritten with the
result when it exists. --apply refuses unless every anchor is in its
pre-flip state and weight_version_key == N-1 (running it twice errors out).
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
PATCH = REPO / "ops" / "v10" / "wvk16_restore_duel.patch"

MODE_OLD = 'crown_mode = "window_best"\n'
MODE_NEW = (
    "# {date} (weight_version_key {a}→{b}, explicit dated operator directive\n"
    "# 2026-09-13 12:10 UTC \"go back to the original delta threshold\"): OFF.\n"
    "# Reign 13 was a re-upload of reign 12's weights (1,026/1,026 tensors\n"
    "# identical, re-sharded 16→2 files) that won window 2515 on z = 0.93 and\n"
    "# a pooled confirmation of +0.000045. Every duel crowns on its own bar\n"
    "# max(k_sigma·SE, δ) again; the window code stays for replay.\n"
    'crown_mode = "duel"\n')
NEAR_OLD = "near_miss_enabled = true\n"
NEAR_NEW = (
    "# {date} (weight_version_key {a}→{b}): OFF with the window rule — one\n"
    "# seeded slice decides, as before 2026-09-11. Code kept for replay.\n"
    "near_miss_enabled = false\n")

ASSERT_LINES = (
    "k_sigma = 2.0\n",
    "min_margin = 0.002\n",
    'min_margin_mode = "fixed"\n',
    "min_z = 0.0\n",
)

BANNER = "# ############################################################################\n"
HISTORY = (
    "# {a}→{b} per-duel crown restored ({date}): explicit dated operator\n"
    "# directive 2026-09-13 12:10 UTC (\"go back to the original delta\n"
    "# threshold, the original system we had more than a day ago; uncrown the\n"
    "# copied model\"). crown_mode window_best → duel and near_miss_enabled\n"
    "# → false: a challenger crowns iff its paired mean(turn_c − turn_k) >\n"
    "# max(k_sigma·SE, min_margin) = max(2·SE, 0.002) on ONE seeded n_turns\n"
    "# slice, plus the unchanged thought-length floor and B gate — the rule of\n"
    "# wvk 3–14. Why: under the window rule reign 13 (chal-00461) was crowned\n"
    "# with reign 12's exact weights (every tensor byte-identical; only the\n"
    "# shard split differed) on a z = 0.93 margin whose confirmation slice\n"
    "# was negative (pooled +0.000045, z 0.08); a second entry with 26\n"
    "# single-element edits was the best candidate of the next window. Under\n"
    "# δ = 0.002 a copy's noise margin crowns at ~1e-4 per duel. Reign 13 is\n"
    "# uncrowned (chal-00461 → rejected_model_copy in history); reign 12\n"
    "# stands. Window / near-miss / decaying-δ code stays behind its knobs so\n"
    "# wvk-15 verdicts replay bit-identically. min(R,G) math unchanged.\n"
    "# Forward-only; min_submission_block unchanged.\n")


def render(src: str, date: str, wvk_to: int) -> str:
    a, b = wvk_to - 1, wvk_to
    fmt = dict(date=date, a=a, b=b)
    for line in ASSERT_LINES:
        if src.count(line) != 1:
            raise SystemExit(f"anchor changed, refusing: expected exactly one {line.strip()!r}")
    for anchor in (MODE_OLD, NEAR_OLD):
        if src.count(anchor) != 1:
            raise SystemExit(f"anchor missing or already flipped: {anchor.strip()!r}")
    key_old = f"weight_version_key = {a}\n"
    if src.count(key_old) != 1:
        raise SystemExit(f"weight_version_key is not {a}")
    out = src.replace(MODE_OLD, MODE_NEW.format(**fmt))
    out = out.replace(NEAR_OLD, NEAR_NEW.format(**fmt))
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
    g.add_argument("--preview", action="store_true",
                   help="write ops/v10/wvk16_restore_duel.patch; toml untouched")
    g.add_argument("--apply", metavar="DATE",
                   help="explicit dated operator directive (YYYY-MM-DD)")
    ap.add_argument("--wvk-to", type=int, help="new weight_version_key (default: current + 1)")
    ap.add_argument("--no-mirror", action="store_true",
                    help="do not overwrite affine/website/code/affine.toml")
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
    print(f"applied wvk {args.wvk_to - 1} -> {args.wvk_to}: crown_mode = duel, "
          f"near_miss_enabled = false ({date}){mirrored}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
