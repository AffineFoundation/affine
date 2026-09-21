"""wvk 14 -> 15: window-best crown in affine.toml.

Operator directive (Jacob Steeves, 2026-09-12 16:39 UTC, confirmed 16:43
UTC "Yes I want to make this update"): the crown is decided per fixed 12 h
window — the king is frozen for the window, the challenger with the best
positive paired margin wins at the window close after a confirmation slice.
This replaces the decaying-δ proposal (kept in code, `min_margin_mode`
stays "fixed"). The mechanism is in the code behind `[duel]` knobs; this
script only turns it on. Flipping the mode changes WHO CROWNS, so it is a
weight_version_key event and runs only with `--apply DATE`.

    python ops/v9/wvk15_toml_edits.py --preview                    # ops/v9/wvk15_window_best.patch
    python ops/v9/wvk15_toml_edits.py --apply 2026-09-12 --wvk-to 15 --mode window_best

What flips:
  crown_mode           "duel" -> "window_best"
  weight_version_key   N-1 -> N   (+ one history paragraph)
Asserted, not edited (already the staged values): crown_window_blocks =
3600, crown_confirm_slice = true, crown_confirm_max = 2,
crown_one_entry_per_hotkey = true, min_margin_mode = "fixed",
min_margin = 0.002, k_sigma = 2.0.

The website mirror `affine/website/code/affine.toml` is overwritten with the
result when it exists (build_llms_txt.py regenerates it anyway).
--apply refuses unless every anchor is in its pre-flip state and
weight_version_key == N-1, so running it twice is a no-op with an error.
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
PATCH = REPO / "ops" / "v9" / "wvk15_window_best.patch"

MODE_OLD = 'crown_mode = "duel"\n'
MODE_NEW = (
    "# {date} (weight_version_key {a}→{b}, explicit dated operator directive\n"
    "# 2026-09-12 16:39/16:43 UTC): ON. The king is frozen per 12 h window;\n"
    "# the best positive margin of the window is crowned at its close after a\n"
    "# confirmation slice. The δ / k_sigma bar below is telemetry from here.\n"
    'crown_mode = "window_best"\n')

ASSERT_LINES = (
    "crown_window_blocks = 3600\n",
    "crown_confirm_slice = true\n",
    "crown_confirm_max = 2\n",
    "crown_one_entry_per_hotkey = true\n",
    'min_margin_mode = "fixed"\n',
    "min_margin = 0.002\n",
    "k_sigma = 2.0\n",
)

BANNER = "# ############################################################################\n"
HISTORY = (
    "# {a}→{b} window-best crown ({date}): the crown is no longer decided duel\n"
    "# by duel. The king is FROZEN for fixed windows of crown_window_blocks =\n"
    "# 3600 chain blocks (12 h; window id = decision block // 3600); every\n"
    "# challenger judged inside a window duels that king; at the window close\n"
    "# the candidate with the largest positive paired margin (one per hotkey,\n"
    "# gates passed) is crowned once ONE fresh n_turns slice keeps its pooled\n"
    "# margin > 0 (crown_confirm_slice; next-best tried up to\n"
    "# crown_confirm_max = 2); nobody confirms → the king stays. The\n"
    "# max(k_sigma·SE, δ) bar is still computed and stamped (duel_rule_wins)\n"
    "# but no longer decides. min(R,G) math, gates, reign chain / payouts\n"
    "# unchanged. Every verdict stamps crown_mode / window_id / decision_block;\n"
    "# every close writes a window_close row (candidates, drops,\n"
    "# confirmations, winner). Forward-only; reign 11 stands;\n"
    "# min_submission_block unchanged.\n")


def render(src: str, date: str, wvk_to: int, mode: str) -> str:
    if mode != "window_best":
        raise SystemExit(f"only --mode window_best is a flip; got {mode!r}")
    a, b = wvk_to - 1, wvk_to
    fmt = dict(date=date, a=a, b=b)
    for line in ASSERT_LINES:
        if src.count(line) != 1:
            raise SystemExit(f"staged value changed, refusing: expected exactly one {line.strip()!r}")
    if src.count(MODE_OLD) != 1:
        raise SystemExit('anchor missing or already flipped: crown_mode = "duel"')
    key_old = f"weight_version_key = {a}\n"
    if src.count(key_old) != 1:
        raise SystemExit(f"weight_version_key is not {a}")
    out = src.replace(MODE_OLD, MODE_NEW.format(**fmt))
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
                   help="write ops/v9/wvk15_window_best.patch; toml untouched")
    g.add_argument("--apply", metavar="DATE",
                   help="explicit dated operator directive (YYYY-MM-DD)")
    ap.add_argument("--wvk-to", type=int, help="new weight_version_key (default: current + 1)")
    ap.add_argument("--mode", default="window_best", choices=["window_best"])
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
    out = render(src, date, args.wvk_to, args.mode)
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
    print(f"applied wvk {args.wvk_to - 1} -> {args.wvk_to}: crown_mode = window_best "
          f"(12 h windows, confirmation slice, one entry per hotkey) ({date}){mirrored}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
