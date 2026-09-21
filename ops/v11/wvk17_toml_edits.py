"""wvk 16 -> 17: wider grounding band + teacher-only reference cap in affine.toml.

Operator directive (Jacob Steeves, 2026-09-14 10:30 UTC: "we can easily go
for one and two today; ... implement and push these once you're confident
the change is ready. Make sure the llms.txt is there and explain."), values
fixed 10:35 UTC from the replay evidence (docs/scoring-today.md
§Recommendation): band_c 2.0 -> 4.0 (band_floor 0.002 unchanged),
ref_max_tokens = 4096 (teacher side only; miners' caps unchanged).

What flips:
  band_c               2.0 -> 4.0      G band half-width w = max(band_c·sd, band_floor)
  ref_max_tokens       (absent) -> 4096  teacher reference sampling budget
                                          (thought + action); miners stay at
                                          max_thought_tokens + max_action_tokens = 1792
  weight_version_key   16 -> 17         (+ one history paragraph)
Asserted, not edited: band_floor = 0.002, max_thought_tokens = 1024,
max_action_tokens = 768, k_sigma = 2.0, min_margin = 0.002,
crown_mode = "duel", near_miss_enabled = false.

    python ops/v11/wvk17_toml_edits.py --preview                     # ops/v11/wvk17_band_refcap.patch
    python ops/v11/wvk17_toml_edits.py --apply 2026-09-14 --wvk-to 17 [--band-c 4.0]

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
PATCH = REPO / "ops" / "v11" / "wvk17_band_refcap.patch"

BAND_OLD = "band_c = 2.0\nband_floor = 0.002\n"
BAND_NEW = (
    "# {date} (weight_version_key {a}→{b}, explicit dated operator directive\n"
    "# 2026-09-14 10:30 UTC): band_c 2.0 → {band_c}. At c = 2 the teacher's OWN\n"
    "# held-out thought fell outside the band its k = 3 references define on\n"
    "# 25.5% of turns — G was noise a quarter of the time. c = {band_c} is the\n"
    "# smallest width with ≥ 90% of held-out teacher thoughts inside (90.3%);\n"
    "# no stored crown or window decision flips, the reign-12 margin moves\n"
    "# z 2.3 → 2.9, the positive control (held-out teacher thought vs base\n"
    "# model) stays at z 4.2 (was 4.64), filler / generic thoughts still lose\n"
    "# at z −8 / −12. band_floor unchanged. Replay worker: docs/scoring-today.md.\n"
    "band_c = {band_c}\n"
    "band_floor = 0.002\n")
CAP_OLD = "max_thought_tokens = 1024\nmax_action_tokens = 768\n"
CAP_NEW = (
    "max_thought_tokens = 1024\n"
    "max_action_tokens = 768\n"
    "# Teacher-only reference budget (thought + action tokens) for the k\n"
    "# reference rollouts. {date} (weight_version_key {a}→{b}, same directive):\n"
    "# 1792 (the miners' shared cap) → 4096. At 1792 the teacher's own\n"
    "# reference hit finish=length on ~21% of deep turns, so those turns had\n"
    "# fewer / truncated references or were dropped (refs < 2): refs per turn\n"
    "# 1.99 → 2.27, turns with a dead R leg 41% → 30% in the replay. Miners'\n"
    "# max_thought_tokens / max_action_tokens are NOT changed by this knob.\n"
    "# Cost: ≈ +15–20% teacher echo compute, +25–40% teacher-side duel wall\n"
    "# time. Absent = the shared cap (pre-wvk-17 replay path).\n"
    "ref_max_tokens = 4096\n")

ASSERT_LINES = (
    "k_sigma = 2.0\n",
    "min_margin = 0.002\n",
    'crown_mode = "duel"\n',
    "near_miss_enabled = false\n",
)

BANNER = "# ############################################################################\n"
HISTORY = (
    "# {a}→{b} wider grounding band + teacher reference cap ({date}): explicit\n"
    "# dated operator directive 2026-09-14 10:30 UTC (\"we can easily go for one\n"
    "# and two today\"). (1) band_c 2.0 → {band_c}: the G leg judges the miner's\n"
    "# thought against the band mu ± max(band_c·sd, band_floor) built from the\n"
    "# teacher's k = 3 reference thoughts; at c = 2 the teacher's OWN held-out\n"
    "# thought was outside that band 25.5% of the time, i.e. G penalised\n"
    "# honest teacher-like thoughts on a quarter of turns. c = {band_c} keeps\n"
    "# 90.3% inside (smallest width ≥ 90%); no stored crown flips; reign-12\n"
    "# margin z 2.3 → 2.9; positive control z 4.2; filler still loses at\n"
    "# z −8 / −12. (2) ref_max_tokens = 4096, TEACHER SIDE ONLY: the k\n"
    "# reference rollouts may run to 4,096 tokens (was the miners' shared\n"
    "# 1,792). At 1,792 the teacher capped ~21% of deep references; refs per\n"
    "# turn 1.99 → 2.27, R-dead turns 41% → 30%. Miners' caps, what miners\n"
    "# emit, min(R,G) math, the crown bar max(2·SE, 0.002), gates and the\n"
    "# reign chain are unchanged. Expected cost: verdicts ~40 → ~50 min.\n"
    "# Forward-only; reign 12 stands; min_submission_block unchanged.\n")


def render(src: str, date: str, wvk_to: int, band_c: str) -> str:
    a, b = wvk_to - 1, wvk_to
    fmt = dict(date=date, a=a, b=b, band_c=band_c)
    for line in ASSERT_LINES:
        if src.count(line) != 1:
            raise SystemExit(f"anchor changed, refusing: expected exactly one {line.strip()!r}")
    for anchor in (BAND_OLD, CAP_OLD):
        if src.count(anchor) != 1:
            raise SystemExit(f"anchor missing or already flipped: {anchor.splitlines()[0]!r}")
    if "ref_max_tokens" in src:
        raise SystemExit("ref_max_tokens already present in the toml")
    key_old = f"weight_version_key = {a}\n"
    if src.count(key_old) != 1:
        raise SystemExit(f"weight_version_key is not {a}")
    out = src.replace(BAND_OLD, BAND_NEW.format(**fmt))
    out = out.replace(CAP_OLD, CAP_NEW.format(**fmt))
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
    ap.add_argument("--band-c", default="4.0", help="new band_c (default 4.0; operator may pick 3.0)")
    ap.add_argument("--no-mirror", action="store_true")
    args = ap.parse_args()
    if not re.fullmatch(r"\d+(\.\d+)?", args.band_c) or float(args.band_c) <= 2.0:
        raise SystemExit("--band-c must be a number > 2.0")
    src = args.toml.read_text()
    if args.wvk_to is None:
        m = re.search(r"^weight_version_key = (\d+)$", src, re.M)
        args.wvk_to = int(m.group(1)) + 1 if m else 0
    date = args.apply or "YYYY-MM-DD"
    if args.apply and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.apply):
        raise SystemExit("--apply needs a YYYY-MM-DD directive date")
    out = render(src, date, args.wvk_to, args.band_c)
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
    print(f"applied wvk {args.wvk_to - 1} -> {args.wvk_to}: band_c = {args.band_c}, "
          f"ref_max_tokens = 4096 ({date}){mirrored}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
