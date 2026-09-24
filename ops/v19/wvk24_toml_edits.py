"""wvk 23 -> 24: sd-meter forfeit floor -12 -> -6 sd. --revert undoes it.

Explicit dated operator directive (Jacob Steeves, 2026-09-23 20:17 UTC, "Lets do
this", quoting the training-speed probe: the floor at -12 carried 48 % of the
per-turn score variance from 2 % of turns).

What flips (--apply DATE):
  [duel.sd_meter].forfeit_sd   -12 -> -6      forfeit floor AND the typ_c floor for a thought with
                                              < content_min_tokens content tokens (same knob)
  weight_version_key           23 -> 24  (+ one history paragraph)
Asserted, not edited: min_margin_sd 0.2, k_sigma 2.0 (x2), n_turns 1000, score_mode sd_min_rga,
thought_rendering as_generated, max_thought_tokens 4096, ref_max_tokens 4864, content_prefix refs_max.
Counterfactual (ops/v19/floor_counterfactual.md, last 30 verdicts): 0 flips, SE x0.89 median
(down to x0.78), z shifts within +-0.5 (one +0.99); genuine valid-turn p1 -4.64 / p0.5 -5.48, so
-6 stays strictly below p1; 0.32 % of valid turns score below -6 (oracle forfeit-seeking gain
0.007 sd/turn = 3.5 % of delta).

    python ops/v19/wvk24_toml_edits.py --preview
    python ops/v19/wvk24_toml_edits.py --apply 2026-09-23
    python ops/v19/wvk24_toml_edits.py --revert 2026-09-23
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
PATCH = REPO / "ops" / "v19" / "wvk24_forfeit_floor.patch"

ASSERT = ("min_margin_sd = 0.2\n", "n_turns = 1000\n", 'score_mode = "sd_min_rga"\n', 'thought_rendering = "as_generated"\n',
          "max_thought_tokens = 4096\n", "ref_max_tokens = 4864\n", 'content_prefix = "refs_max"\n')
OLD = "forfeit_sd = -12\n"
NEW = ("# {date} (wvk {a}→{b}, explicit dated operator directive 2026-09-23 20:17 UTC \"Lets\n"
       "# do this\"): −12 → −6. At −12 the 2 % of forfeited side-turns carried 48 % of the\n"
       "# per-turn score variance (training-speed probe); −6 is still strictly below\n"
       "# the genuine valid-turn p1 (−4.64) and p0.5 (−5.48): 0.32 % of valid turns\n"
       "# score under it, oracle forfeit-seeking gain 0.007 sd/turn (3.5 % of δ).\n"
       "# Counterfactual on the last 30 verdicts: 0 flips, SE ×0.89 median. Also the\n"
       "# typ_c floor for thoughts with < content_min_tokens content tokens.\n"
       "forfeit_sd = -6\n")
BANNER = "# ############################################################################\n"
HISTORY = (
    "# {a}→{b} forfeit floor −12 → −6 sd ({date}): explicit dated operator directive\n"
    "# 2026-09-23 20:17 UTC (\"Lets do this\", on the training-speed probe: 2.0 % of\n"
    "# side-turns at −12 carried 48 % of the per-turn score variance). −6 stays\n"
    "# strictly below the genuine valid-turn p1 (−4.64; p0.5 −5.48); 0.32 % of\n"
    "# valid turns score below it (oracle forfeit-seeking gain 0.007 sd/turn =\n"
    "# 3.5 % of δ; at −7: 0.19 % / 0.0045). Counterfactual on the last 30 verdicts:\n"
    "# 0 decision flips, SE ×0.89 median (×0.78 at best), z shifts within ±0.5\n"
    "# (one +0.99). A 2 % forfeit gap now costs ≈ 0.09 sd (half a δ; was one δ).\n"
    "# Nothing else changes: δ 0.2, k_sigma 2, n_turns 1000, caps, rendering,\n"
    "# content_prefix. Forward-only, reign 21 stands, min_submission_block\n"
    "# unchanged.\n")
REVERT_HISTORY = ("# {a}→{b} ROLLBACK to the wvk-23 floor ({date}): explicit operator-directed\n"
                  "# revert (control z sign flip). forfeit_sd −6 → −12. wvk-24 verdicts stand.\n")


def _check(src, line):
    if src.count(line) != 1:
        raise SystemExit(f"anchor {line.strip()!r}: expected exactly one occurrence, got {src.count(line)}")


def _history(out, paragraph):
    idx = out.find(BANNER)
    if idx < 0 or out.find(BANNER, idx + 1) < 0:
        raise SystemExit("anchor missing: weight_version_key banner")
    head, tail = out[:idx], out[idx:]
    if not head.endswith("#\n"):
        raise SystemExit("unexpected text above the banner")
    return head[:-2] + paragraph + "#\n" + tail


def render(src, date, a=23, b=24):
    fmt = dict(date=date, a=a, b=b)
    for line in ASSERT + (OLD, f"weight_version_key = {a}\n"):
        _check(src, line)
    if src.count("k_sigma = 2.0\n") != 2:
        raise SystemExit("k_sigma = 2.0 expected twice")
    out = src.replace(OLD, NEW.format(**fmt)).replace(f"weight_version_key = {a}\n", f"weight_version_key = {b}\n")
    return _history(out, HISTORY.format(**fmt))


def render_revert(src, date):
    for line in ("forfeit_sd = -6\n", "weight_version_key = 24\n"):
        _check(src, line)
    out = src.replace("forfeit_sd = -6\n", "forfeit_sd = -12\n").replace("weight_version_key = 24\n", "weight_version_key = 23\n")
    return _history(out, REVERT_HISTORY.format(date=date, a=24, b=23))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--toml", type=Path, default=TOML)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--preview", action="store_true"); g.add_argument("--apply", metavar="DATE"); g.add_argument("--revert", metavar="DATE")
    ap.add_argument("--no-mirror", action="store_true")
    a = ap.parse_args()
    src = a.toml.read_text(); date = a.apply or a.revert or "YYYY-MM-DD"
    if (a.apply or a.revert) and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        raise SystemExit("--apply/--revert need a YYYY-MM-DD directive date")
    out = render_revert(src, date) if a.revert else render(src, date)
    what = "REVERT wvk 24 -> 23 (forfeit_sd -12)" if a.revert else "wvk 23 -> 24: forfeit_sd -6"
    if a.preview:
        PATCH.write_text("".join(difflib.unified_diff(src.splitlines(True), out.splitlines(True), "a/affine/affine.toml", "b/affine/affine.toml")))
        print(f"wrote {PATCH} ({what})"); return 0
    a.toml.write_text(out)
    mirrored = ""
    if not a.no_mirror and MIRROR.exists() and a.toml.resolve() == TOML.resolve():
        shutil.copyfile(a.toml, MIRROR); mirrored = f"; mirrored to {MIRROR.relative_to(REPO)}"
    print(f"applied {what} ({date}){mirrored}", file=sys.stderr); return 0


if __name__ == "__main__":
    raise SystemExit(main())
