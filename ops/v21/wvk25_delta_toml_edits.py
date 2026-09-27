"""wvk 24 -> 25: sd-meter crown floor δ 0.20 -> 0.10 sd. --revert undoes it.

Explicit dated operator directive (Jacob Steeves, 2026-09-27 08:22 UTC, "Lower the
validator margin to 0.1"). Its own fork; the GLM / 262k / scoring bundle noticed
for 2026-09-30 14:00 UTC becomes wvk 26.

What flips (--apply DATE):
  [duel.sd_meter].min_margin_sd   0.2 -> 0.1
  weight_version_key              24 -> 25  (+ one history paragraph)
Asserted, not edited: forfeit_sd -6, k_sigma 2.0 (x2), n_turns 1000, score_mode sd_min_rga,
thought_rendering as_generated, max_thought_tokens 4096, ref_max_tokens 4864,
content_prefix refs_max, ref_min_content 10, typ_min_refs 2.
Counterfactual (108 sd-meter verdicts since wvk 22, each vs its own then-king): 8 more crowns
(chal-00613/631/643/649/651/652/653/655, margins +0.11..+0.20, z 2.5..4.1), the 7 real ones
stand, chal-00687 (+0.073) stays under. SE median 0.047 at n~1000 -> δ 0.10 ≈ 2.1 SE (was 4.2);
δ binds on 59/108 verdicts (was 94/108).

    python ops/v21/wvk25_delta_toml_edits.py --preview
    python ops/v21/wvk25_delta_toml_edits.py --apply 2026-09-27
    python ops/v21/wvk25_delta_toml_edits.py --revert 2026-09-27
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
PATCH = REPO / "ops" / "v21" / "wvk25_delta.patch"

ASSERT = ("forfeit_sd = -6\n", "n_turns = 1000\n", 'score_mode = "sd_min_rga"\n', 'thought_rendering = "as_generated"\n',
          "max_thought_tokens = 4096\n", "ref_max_tokens = 4864\n", 'content_prefix = "refs_max"\n',
          "ref_min_content = 10\n", "typ_min_refs = 2\n")
OLD = "min_margin_sd = 0.2\n"
NEW = ("# {date} (wvk {a}→{b}, explicit dated operator directive 2026-09-27 08:22 UTC \"Lower\n"
       "# the validator margin to 0.1\"): 0.2 → 0.1. At the live noise (SE median 0.047 at\n"
       "# n ≈ 1000) δ = 0.10 is ≈ 2.1 SE — the bar the 2026-08-21/22 δ-at-noise experiment\n"
       "# reverted after winner's-curse crown churn; the operator chose it knowingly.\n"
       "min_margin_sd = 0.1\n")
BANNER = "# ############################################################################\n"
HISTORY = (
    "# {a}→{b} crown floor δ 0.20 → 0.10 sd ({date}): explicit dated operator directive\n"
    "# 2026-09-27 08:22 UTC (Jacob Steeves, \"Lower the validator margin to 0.1\"). Own\n"
    "# fork; the GLM-5.3-Flash teacher / 262k / scoring bundle noticed for 2026-09-30\n"
    "# 14:00 UTC becomes wvk 26. Counterfactual on the 108 sd-meter verdicts since wvk\n"
    "# 22 (each vs its own then-king): 8 more crowns (chal-00613/631/643/649/651/652/\n"
    "# 653/655; margins +0.11…+0.20, z 2.5…4.1), the 7 real crowns stand, chal-00687\n"
    "# (+0.073, reign 22 by operator crown) stays under. Known risk, accepted by the\n"
    "# operator: SE at n ≈ 1000 is median 0.047 (p10 0.030 / p90 0.110), so δ 0.10 is\n"
    "# ≈ 2.1 SE (0.9–3.3 across duels; δ 0.20 was 4.2) and binds on 59/108 verdicts\n"
    "# instead of 94/108 — the noisier half of duels is decided by the 2σ test alone.\n"
    "# That is the 2026-08-21/22 configuration (δ 0.002 → 0.001 at the v4 noise\n"
    "# floor: 4 near-noise crowns in 18 h, the winners' measured score drifting down\n"
    "# across the chain = winner's-curse churn; reverted wvk 8→9 the next day). If\n"
    "# churn returns, `--revert` puts 0.2 back as its own fork. Nothing else changes:\n"
    "# k_sigma 2, n_turns 1000, forfeit −6, caps, rendering, content_prefix, empty-\n"
    "# ref rule. Forward-only, reign 22 stands, min_submission_block unchanged.\n")
REVERT_HISTORY = ("# {a}→{b} ROLLBACK of the δ 0.10 floor ({date}): explicit operator-directed revert\n"
                  "# (crown churn). min_margin_sd 0.1 → 0.2. wvk-25 verdicts stand.\n")


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


def render(src, date, a=24, b=25):
    fmt = dict(date=date, a=a, b=b)
    for line in ASSERT + (OLD, f"weight_version_key = {a}\n"):
        _check(src, line)
    if src.count("k_sigma = 2.0\n") != 2:
        raise SystemExit("k_sigma = 2.0 expected twice")
    out = src.replace(OLD, NEW.format(**fmt)).replace(f"weight_version_key = {a}\n", f"weight_version_key = {b}\n")
    return _history(out, HISTORY.format(**fmt))


def render_revert(src, date):
    for line in ("min_margin_sd = 0.1\n", "weight_version_key = 25\n"):
        _check(src, line)
    out = src.replace("min_margin_sd = 0.1\n", "min_margin_sd = 0.2\n").replace("weight_version_key = 25\n", "weight_version_key = 24\n")
    return _history(out, REVERT_HISTORY.format(date=date, a=25, b=24))


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
    what = "REVERT wvk 25 -> 24 (min_margin_sd 0.2)" if a.revert else "wvk 24 -> 25: min_margin_sd 0.1"
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
