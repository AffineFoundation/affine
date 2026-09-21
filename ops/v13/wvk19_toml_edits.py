"""wvk 18 -> 19: a crown must win twice — confirmation slice for first-slice passes.

Operator directive (Jacob Steeves, 2026-09-16 11:10 UTC, "lets do B").

What flips:
  confirmation_required   (absent = false) -> true
      a duel that clears max(k_sigma·SE, δ) + gates is a CANDIDATE; the
      validator runs one more independent n_turns slice (seed
      blake2b(block_hash ‖ hotkey ‖ "|slice1"), turns disjoint, fresh teacher
      refs, warm engines, ~40 min) and crowns only if that slice's own margin
      is > 0 AND the pooled margin over both slices clears
      max(k_sigma·SE_pooled, δ). Otherwise: rejection_reason
      "confirmation_failed", a loss, king stands, slot consumed.
  weight_version_key      18 -> 19  (+ one history paragraph)
Asserted, not edited: k_sigma = 2.0, min_margin = 0.002, crown_mode = "duel",
near_miss_enabled = false, max_thought_tokens = 2048, ref_max_tokens = 4096.

    python ops/v13/wvk19_toml_edits.py --preview                 # ops/v13/wvk19_confirmation.patch
    python ops/v13/wvk19_toml_edits.py --apply 2026-09-16 --wvk-to 19
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
PATCH = REPO / "ops" / "v13" / "wvk19_confirmation.patch"

ANCHOR_OLD = "crown_one_entry_per_hotkey = true\n"
ANCHOR_NEW = (
    "crown_one_entry_per_hotkey = true\n"
    "# {date} (weight_version_key {a}→{b}, explicit dated operator directive\n"
    "# 2026-09-16 11:10 UTC \"lets do B\"): a crown must win twice. A duel that\n"
    "# clears max(k_sigma·SE, δ) + gates is a candidate, not a king: the\n"
    "# validator scores ONE more independent n_turns slice against the same\n"
    "# king (seed blake2b(block_hash ‖ hotkey ‖ \"|slice1\"), turns disjoint from\n"
    "# the first slice, fresh teacher references, warm engines, ~40 min) and\n"
    "# crowns only if that slice's own paired margin is > 0 AND the pooled\n"
    "# margin over both slices clears max(k_sigma·SE_pooled, δ). Otherwise the\n"
    "# verdict is a loss (rejection_reason confirmation_failed): king stands,\n"
    "# slot consumed, nothing re-queued. Evidence: all 15 crowns since the\n"
    "# wvk-10 reset were single-slice passes (3.4% of 447 scored verdicts);\n"
    "# the only two with a stored second slice (the revoked window crowns\n"
    "# chal-00454 / chal-00461) fail this test. A null challenger crowns by\n"
    "# noise on ≈0.5% of attempts today (δ ≈ 2.6·SE) → ≈0.009% with the\n"
    "# confirmation (Monte Carlo at SE 0.00078). Verdicts stamp\n"
    "# duel_params.confirmation_required and verdict.confirmation {{seed, n,\n"
    "# margin, se, z, pooled_margin, pooled_se, pooled_z, bar, passed}}.\n"
    "# false = pre-wvk-19 replay (crown on the first slice).\n"
    "confirmation_required = true\n")

ASSERT_LINES = (
    "k_sigma = 2.0\n",
    "min_margin = 0.002\n",
    'crown_mode = "duel"\n',
    "near_miss_enabled = false\n",
    "max_thought_tokens = 2048\n",
    "ref_max_tokens = 4096\n",
)

BANNER = "# ############################################################################\n"
HISTORY = (
    "# {a}→{b} confirmation slice for crowns ({date}): explicit dated operator\n"
    "# directive 2026-09-16 11:10 UTC (\"lets do B\"). The per-duel rule stays\n"
    "# (paired margin > max(k_sigma·SE, δ), k_sigma 2.0, δ 0.002, thought-length\n"
    "# floor, B gate) but a pass no longer crowns: the validator runs a second\n"
    "# independent 1,300-turn slice (fresh seed from the reveal hash, fresh\n"
    "# teacher refs, same warm engines, ~40 min) and crowns only if that\n"
    "# slice's margin is > 0 AND the pooled margin over both slices clears\n"
    "# max(k_sigma·SE_pooled, δ); else confirmation_failed = a loss, king\n"
    "# stands. Why: every one of the 15 crowns since the wvk-10 reset rested\n"
    "# on one slice; the two crowns that did get a second slice (wvk-15\n"
    "# window rule; both later revoked) had it come back ≤ 0. Noise crowns\n"
    "# fall from ≈0.5% to ≈0.009% per attempt at today's SE; an honest\n"
    "# improver waits ~40 min. min(R,G) math, caps, gates, reign chain\n"
    "# unchanged; wvk ≤ 18 verdicts replay through the knob's false default.\n"
    "# Forward-only; reign 13 stands; min_submission_block unchanged.\n")


def render(src: str, date: str, wvk_to: int) -> str:
    a, b = wvk_to - 1, wvk_to
    fmt = dict(date=date, a=a, b=b)
    for line in ASSERT_LINES:
        if src.count(line) != 1:
            raise SystemExit(f"anchor changed, refusing: expected exactly one {line.strip()!r}")
    if src.count(ANCHOR_OLD) != 1:
        raise SystemExit("anchor missing: crown_one_entry_per_hotkey = true")
    if "confirmation_required" in src:
        raise SystemExit("confirmation_required already present in the toml")
    key_old = f"weight_version_key = {a}\n"
    if src.count(key_old) != 1:
        raise SystemExit(f"weight_version_key is not {a}")
    out = src.replace(ANCHOR_OLD, ANCHOR_NEW.format(**fmt))
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
    print(f"applied wvk {args.wvk_to - 1} -> {args.wvk_to}: confirmation_required = true "
          f"({date}){mirrored}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
