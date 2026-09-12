"""Flip the decaying crown margin in affine.toml (wvk 14 -> 15).

Operator directive (Jacob Steeves, 2026-09-12 15:27 UTC): "reset-to-cap at
each crown — let's do that and launch this + the min Z of half the margin."
The mechanism is already in the code behind `[duel]` knobs (PR #16,
`affine/affine/score.py::MarginSchedule`); this script only turns it on.
Flipping the mode CHANGES WHICH MARGINS CROWN, so it is a weight_version_key
event and runs only with `--apply DATE` naming the directive date.

    python ops/v9/decaying_delta_toml_edits.py --preview [--variant minz|floor]
    python ops/v9/decaying_delta_toml_edits.py --apply 2026-09-12 --wvk-to 15 --variant minz
    python ops/v9/decaying_delta_toml_edits.py --apply 2026-09-12 --wvk-to 15 --variant floor

Two variants of "the min Z of half the margin" (one of them is picked by
the operator; the rest of the flip is identical):

  minz   min_z = 2.5, min_margin_floor stays 0.0001   (the doc's recommendation:
         replay 14 crowns / 0 ties / 0 real crowns lost)
  floor  min_margin_floor = 0.001 (= half the 0.002 cap), min_z stays 0.0

`--min-z X` / `--floor Y` override either number for any other combination.

What flips (the why lives next to each knob in affine.toml):
  min_margin_mode             "fixed" -> "decay"
  min_margin_double_on_crown  true -> false     reset to the cap at every crown
  near_miss_window_mode       "absolute" -> "bar"
  min_z / min_margin_floor    per variant
  weight_version_key          N-1 -> N          (+ one history paragraph)
  min_margin_peak_cap 0.002 / min_margin_decay_hours 48.0 / shape "linear"
  are already the staged values and are asserted, not edited.

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
PATCH_DIR = REPO / "ops" / "v9"

VARIANTS = {
    "minz": dict(min_z=2.5, floor=0.0001),
    "floor": dict(min_z=0.0, floor=0.001),
}

MODE_OLD = 'min_margin_mode = "fixed"\n'
MODE_NEW = (
    "# {date} (weight_version_key {a}→{b}, explicit dated operator directive\n"
    "# 2026-09-12 15:27 UTC, \"reset-to-cap at each crown — let's do that and\n"
    "# launch this\"): ON. δ resets to min_margin_peak_cap at every crown and\n"
    "# decays to min_margin_floor over min_margin_decay_hours, block-clocked.\n"
    'min_margin_mode = "decay"\n')

DOUBLE_OLD = "min_margin_double_on_crown = true\n"
DOUBLE_NEW = (
    "# {date}: false = reset to the cap at every crown (operator choice; the\n"
    "# literal doubling collapses to 2·floor after one full cycle).\n"
    "min_margin_double_on_crown = false\n")

FLOOR_OLD = "min_margin_floor = 0.0001\n"
FLOOR_NEW = (
    "# {date} (weight_version_key {a}→{b}): floor = half the cap, the hard δ\n"
    "# floor that stands in for a minimum z (\"the min Z of half the margin\").\n"
    "min_margin_floor = {floor:g}\n")

MINZ_OLD = "min_z = 0.0\n"
MINZ_NEW = (
    "# {date} (weight_version_key {a}→{b}, explicit dated operator directive):\n"
    "# ON at {min_z:g} — the safeguard that pairs with the decaying δ; a\n"
    "# statistical tie cannot crown once δ is at the floor.\n"
    "min_z = {min_z:g}\n")

WINDOW_OLD = 'near_miss_window_mode = "absolute"\n'
WINDOW_NEW = (
    "# {date}: \"bar\" — the second-slice window follows this slice's own\n"
    "# crown bar now that δ moves (drop-in at δ = 0.002; see above).\n"
    'near_miss_window_mode = "bar"\n')

ASSERT_LINES = (
    "min_margin_peak_cap = 0.002\n",
    "min_margin_decay_hours = 48.0\n",
    'min_margin_decay_shape = "linear"\n',
    "min_margin = 0.002\n",
)

BANNER = "# ############################################################################\n"
HISTORY_COMMON = (
    "# {a}→{b} decaying crown margin ({date}): δ is no longer a fixed 0.002.\n"
    "# At every crown δ resets to min_margin_peak_cap (0.002) and falls\n"
    "# linearly to min_margin_floor ({floor:g}) over min_margin_decay_hours\n"
    "# (48 h), clocked in blocks since the crown block (decision block −\n"
    "# crown block; 12 s/block); it stays at the floor until the next crown.\n")
HISTORY_MINZ = (
    "# A crown also needs z = margin/SE ≥ min_z ({min_z:g}), whatever δ is.\n")
HISTORY_TAIL = (
    "# The near-miss second slice triggers on (0.5·bar, 1.5·bar) of the\n"
    "# slice's own bar (near_miss_window_mode = \"bar\"). Crown formula\n"
    "# max(k_sigma·SE, δ) + gates and the min(R,G) math are unchanged. Every\n"
    "# verdict stamps duel_params.min_margin_effective / min_margin_mode /\n"
    "# crown_block / decision_block / blocks_since_crown. Forward-only; reign\n"
    "# stands; the sitting king's cycle is read from its reveal block.\n")


def _replace_once(src: str, old: str, new: str, what: str) -> str:
    if src.count(old) != 1:
        raise SystemExit(f"anchor missing or already flipped: {what} ({old.strip()!r})")
    return src.replace(old, new)


def render(src: str, date: str, wvk_to: int, min_z: float, floor: float) -> str:
    a, b = wvk_to - 1, wvk_to
    fmt = dict(date=date, a=a, b=b, min_z=min_z, floor=floor)
    for line in ASSERT_LINES:
        if src.count(line) != 1:
            raise SystemExit(f"staged value changed, refusing: expected exactly one {line.strip()!r}")
    key_old = f"weight_version_key = {a}\n"
    if src.count(key_old) != 1:
        raise SystemExit(f"weight_version_key is not {a}")
    out = _replace_once(src, MODE_OLD, MODE_NEW.format(**fmt), "min_margin_mode")
    out = _replace_once(out, DOUBLE_OLD, DOUBLE_NEW.format(**fmt), "min_margin_double_on_crown")
    out = _replace_once(out, WINDOW_OLD, WINDOW_NEW.format(**fmt), "near_miss_window_mode")
    if floor != 0.0001:
        out = _replace_once(out, FLOOR_OLD, FLOOR_NEW.format(**fmt), "min_margin_floor")
    elif out.count(FLOOR_OLD) != 1:
        raise SystemExit("anchor missing: min_margin_floor = 0.0001")
    if min_z > 0:
        out = _replace_once(out, MINZ_OLD, MINZ_NEW.format(**fmt), "min_z")
    elif out.count(MINZ_OLD) != 1:
        raise SystemExit("anchor missing: min_z = 0.0")
    out = out.replace(key_old, f"weight_version_key = {b}\n")
    idx = out.find(BANNER)
    if idx < 0 or out.find(BANNER, idx + 1) < 0:
        raise SystemExit("anchor missing: weight_version_key banner")
    head, tail = out[:idx], out[idx:]
    if not head.endswith("#\n"):
        raise SystemExit("unexpected text above the banner")
    history = HISTORY_COMMON.format(**fmt)
    if min_z > 0:
        history += HISTORY_MINZ.format(**fmt)
    history += HISTORY_TAIL.format(**fmt)
    return head[:-2] + history + "#\n" + tail


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--toml", type=Path, default=TOML)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--preview", action="store_true",
                   help="write ops/v9/wvk15_<variant>.patch; toml untouched")
    g.add_argument("--apply", metavar="DATE",
                   help="explicit dated operator directive (YYYY-MM-DD)")
    ap.add_argument("--wvk-to", type=int, help="new weight_version_key (default: current + 1)")
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="minz",
                    help="minz: min_z = 2.5 (default); floor: min_margin_floor = 0.001")
    ap.add_argument("--min-z", type=float, help="override the variant's min_z")
    ap.add_argument("--floor", type=float, help="override the variant's min_margin_floor")
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
    v = VARIANTS[args.variant]
    min_z = v["min_z"] if args.min_z is None else args.min_z
    floor = v["floor"] if args.floor is None else args.floor
    if min_z < 0 or not (0.0 < floor <= 0.002):
        raise SystemExit(f"bad values: min_z={min_z} floor={floor}")
    out = render(src, date, args.wvk_to, min_z, floor)
    if args.preview:
        patch = PATCH_DIR / f"wvk{args.wvk_to}_{args.variant}.patch"
        patch.write_text("".join(difflib.unified_diff(
            src.splitlines(True), out.splitlines(True),
            "a/affine/affine.toml", "b/affine/affine.toml")))
        print(f"wrote {patch}")
        return 0
    args.toml.write_text(out)
    if not args.no_mirror and MIRROR.exists() and args.toml.resolve() == TOML.resolve():
        shutil.copyfile(args.toml, MIRROR)
        mirrored = f"; mirrored to {MIRROR.relative_to(REPO)}"
    else:
        mirrored = ""
    print(f"applied v9 decaying margin: weight_version_key {args.wvk_to - 1} -> {args.wvk_to}, "
          f"min_margin_mode = decay (reset-to-cap, 48 h linear, floor {floor:g}), "
          f"min_z = {min_z:g}, near_miss_window_mode = bar ({date}){mirrored}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
