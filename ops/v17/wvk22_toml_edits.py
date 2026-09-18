"""wvk 21 -> 22: the sd-meter min(z_R, typ_c, z_A) becomes the scoring rule;
n_turns 1300 -> 1000. Also the one-command revert (--revert) for the rollback.

Explicit dated operator directive (Jacob Steeves, 2026-09-18 10:04 UTC "I
like it. And I want to ship it"; 10:25 UTC "lets reduce the turns to 1000 for
scoring speed"; the go to flip is relayed by the coordinator and MUST be
given before --apply is run).

What flips (--apply DATE --wvk-to 22 --delta-sd X --forfeit-sd Y):
  score_mode          "min_rg" -> "sd_min_rga"
  n_turns             1300 -> 1000
  sd_meter.min_margin_sd   0.0 -> X      (δ in teacher-sd units)
  sd_meter.forfeit_sd      -2.4 -> Y     (forfeit floor in sd units)
  weight_version_key  21 -> 22  (+ one history paragraph)
Asserted, not edited: k_sigma = 2.0, min_margin = 0.002 (kept for wvk ≤ 21
replay; unused by the new rule), band_c = 4.0, band_floor = 0.002 (same),
forfeit_turn_score = -0.1 (same), tau = 0.03, thought_cap_ratio = 1.25,
ref_max_tokens = 4096, require_think_close = true.

--revert DATE: score_mode back to "min_rg", n_turns 1300, wvk 22 -> 21,
plus a history paragraph naming the rollback. Explicit operator-directed
revert only (the rollback rule in docs/wvk22-plan.md §3).

    python ops/v17/wvk22_toml_edits.py --preview --delta-sd 0.10 --forfeit-sd -2.4
    python ops/v17/wvk22_toml_edits.py --apply 2026-09-18 --wvk-to 22 --delta-sd 0.10 --forfeit-sd -2.4
    python ops/v17/wvk22_toml_edits.py --revert 2026-09-18
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
PATCH = REPO / "ops" / "v17" / "wvk22_sd_meter.patch"

ASSERT_LINES = (
    "min_margin = 0.002\n",
    "band_c = 4.0\n",
    "band_floor = 0.002\n",
    "forfeit_turn_score = -0.1\n",
    "tau = 0.03\n",
    "thought_cap_ratio = 1.25\n",
    "ref_max_tokens = 4096\n",
    "require_think_close = true\n",
)

SCORE_OLD = 'score_mode = "min_rg"\n'
SCORE_NEW = (
    '# {date} (weight_version_key {a}→{b}, explicit dated operator directive\n'
    '# 2026-09-18 10:04 UTC "I like it. And I want to ship it"): the sd-meter.\n'
    '# turn = min(z_R, typ_c, z_A) in teacher-sd units — see [duel.sd_meter]\n'
    '# for the definition and knobs. min_rg stays replayable for wvk 10–21.\n'
    'score_mode = "sd_min_rga"\n')
N_OLD = "n_turns = 1300\n"
N_NEW = (
    "# {date} (wvk {a}→{b}, same directive, 10:25 UTC \"lets reduce the turns to\n"
    "# 1000 for scoring speed\"): 1300 → 1000. SE × 1.14; δ_sd is set against\n"
    "# the measured SE so the crown bar keeps its ratio.\n"
    "n_turns = 1000\n")
DELTA_OLD = "min_margin_sd = 0.0\n"
FORFEIT_OLD = "forfeit_sd = -2.4\n"

BANNER = "# ############################################################################\n"
HISTORY = (
    "# {a}→{b} sd-meter + 1,000-turn slices ({date}): explicit dated operator\n"
    "# directive 2026-09-18 (10:04 UTC \"I like it. And I want to ship it\";\n"
    "# 10:25 UTC \"lets reduce the turns to 1000 for scoring speed\"; go to flip\n"
    "# relayed by the coordinator). score_mode min_rg → sd_min_rga: turn =\n"
    "# min(z_R, typ_c, z_A) — the largest standardised deviation of the reply\n"
    "# from the teacher's own samples across thought typicality (content\n"
    "# tokens, |prefix lift| > 1 nat), thought→action and action←thought, in\n"
    "# teacher-sd units (LOO anchors per turn, σ pooled per dialect). n_turns\n"
    "# 1300 → 1000. δ = {delta} sd, k_sigma 2.0, forfeit {forfeit} sd (calibrated\n"
    "# on the shadow read to keep today's crown rate: δ ≈ 1.5× the 2σ bar,\n"
    "# floor ≈ the live −0.1's sd position). band_c / band_floor / min_margin /\n"
    "# forfeit_turn_score kept for wvk ≤ 21 replay, unused by the new rule.\n"
    "# Forward-only: reign 14 stands, no re-verdicts, min_submission_block\n"
    "# unchanged. Notice: llms.txt \"Upcoming change: wvk 22\" + Discord\n"
    "# 2026-09-18 10:46 UTC; shadow read since 10:41 UTC.\n")
REVERT_HISTORY = (
    "# {a}→{b} ROLLBACK to the wvk-21 settings ({date}): explicit operator-\n"
    "# directed revert under the rule of docs/wvk22-plan.md §3 (first wvk-22\n"
    "# verdicts failed the sanity criteria). score_mode sd_min_rga → min_rg,\n"
    "# n_turns 1000 → 1300; [duel.sd_meter] stays as shadow telemetry.\n"
    "# Verdicts judged under wvk 22 in between stand (forward-only both ways).\n")


def _check(src: str, line: str, what: str = "") -> None:
    if src.count(line) != 1:
        raise SystemExit(f"anchor {what or line.strip()!r}: expected exactly one occurrence")


def render(src: str, date: str, wvk_to: int, delta: float, forfeit: float) -> str:
    a, b = wvk_to - 1, wvk_to
    fmt = dict(date=date, a=a, b=b, delta=f"{delta:g}", forfeit=f"{forfeit:g}")
    for line in ASSERT_LINES:
        _check(src, line)
    if src.count("k_sigma = 2.0\n") != 2:      # [duel] and [duel.sd_meter]
        raise SystemExit("anchor 'k_sigma = 2.0': expected twice ([duel] + [duel.sd_meter])")
    for line in (SCORE_OLD, N_OLD, DELTA_OLD, FORFEIT_OLD, f"weight_version_key = {a}\n"):
        _check(src, line)
    out = src.replace(SCORE_OLD, SCORE_NEW.format(**fmt))
    out = out.replace(N_OLD, N_NEW.format(**fmt))
    out = out.replace(DELTA_OLD, f"min_margin_sd = {delta:g}\n")
    out = out.replace(FORFEIT_OLD, f"forfeit_sd = {forfeit:g}\n")
    out = out.replace(f"weight_version_key = {a}\n", f"weight_version_key = {b}\n")
    return _history(out, HISTORY.format(**fmt))


def render_revert(src: str, date: str) -> str:
    _check(src, 'score_mode = "sd_min_rga"\n')
    _check(src, "n_turns = 1000\n")
    _check(src, "weight_version_key = 22\n")
    out = src.replace('score_mode = "sd_min_rga"\n', 'score_mode = "min_rg"\n')
    out = out.replace("n_turns = 1000\n", "n_turns = 1300\n")
    out = out.replace("weight_version_key = 22\n", "weight_version_key = 21\n")
    return _history(out, REVERT_HISTORY.format(date=date, a=22, b=21))


def _history(out: str, paragraph: str) -> str:
    idx = out.find(BANNER)
    if idx < 0 or out.find(BANNER, idx + 1) < 0:
        raise SystemExit("anchor missing: weight_version_key banner")
    head, tail = out[:idx], out[idx:]
    if not head.endswith("#\n"):
        raise SystemExit("unexpected text above the banner")
    return head[:-2] + paragraph + "#\n" + tail


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--toml", type=Path, default=TOML)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--preview", action="store_true")
    g.add_argument("--apply", metavar="DATE")
    g.add_argument("--revert", metavar="DATE")
    ap.add_argument("--wvk-to", type=int, default=22)
    ap.add_argument("--delta-sd", type=float)
    ap.add_argument("--forfeit-sd", type=float)
    ap.add_argument("--no-mirror", action="store_true")
    args = ap.parse_args()
    src = args.toml.read_text()
    date = args.apply or args.revert or "YYYY-MM-DD"
    if (args.apply or args.revert) and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        raise SystemExit("--apply/--revert need a YYYY-MM-DD directive date")
    if args.revert:
        out = render_revert(src, date)
        what = "REVERT wvk 22 -> 21 (score_mode min_rg, n_turns 1300)"
    else:
        if args.delta_sd is None or args.forfeit_sd is None:
            raise SystemExit("--delta-sd and --forfeit-sd are required (final calibrated values)")
        if args.delta_sd < 0 or args.forfeit_sd > 0:
            raise SystemExit("delta_sd must be >= 0 and forfeit_sd <= 0")
        out = render(src, date, args.wvk_to, args.delta_sd, args.forfeit_sd)
        what = (f"wvk {args.wvk_to - 1} -> {args.wvk_to}: score_mode sd_min_rga, n_turns 1000, "
                f"δ {args.delta_sd:g} sd, forfeit {args.forfeit_sd:g} sd")
    if args.preview:
        PATCH.write_text("".join(difflib.unified_diff(
            src.splitlines(True), out.splitlines(True),
            "a/affine/affine.toml", "b/affine/affine.toml")))
        print(f"wrote {PATCH} ({what})")
        return 0
    args.toml.write_text(out)
    mirrored = ""
    if not args.no_mirror and MIRROR.exists() and args.toml.resolve() == TOML.resolve():
        shutil.copyfile(args.toml, MIRROR)
        mirrored = f"; mirrored to {MIRROR.relative_to(REPO)}"
    print(f"applied {what} ({date}){mirrored}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
