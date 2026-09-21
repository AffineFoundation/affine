"""wvk 19 -> 20: teacher-relative miner thought cap.

Operator directive (Jacob Steeves, 2026-09-16 14:31 UTC, "can we ship this
now?" -> yes). Evidence: docs/teacher-gated-cap.md.

What flips:
  thought_cap_ratio   (absent = 0 = fixed cap) -> 1.25
      per turn the miner may think up to cap_T = max(max_thought_tokens,
      floor(1.25 × L_T)) tokens, L_T = the longest VALID teacher reference
      thought on that turn in teacher-tokenizer tokens (Qwen/Qwen3.8-27B).
      Both sides get the same cap_T (shared references). Action cap 768 and
      the teacher's ref_max_tokens 4096 unchanged.
  weight_version_key  19 -> 20  (+ one history paragraph)
Asserted, not edited: max_thought_tokens = 2048, max_action_tokens = 768,
ref_max_tokens = 4096, confirmation_required = true, crown_mode = "duel",
k_sigma = 2.0, min_margin = 0.002.

    python ops/v15/wvk20_toml_edits.py --preview                 # ops/v15/wvk20_teacher_cap.patch
    python ops/v15/wvk20_toml_edits.py --apply 2026-09-16 --wvk-to 20
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
PATCH = REPO / "ops" / "v15" / "wvk20_teacher_cap.patch"

ANCHOR_OLD = "max_thought_tokens = 2048\nmax_action_tokens = 768\n"
ANCHOR_NEW = (
    "max_thought_tokens = 2048\n"
    "max_action_tokens = 768\n"
    "# {date} (weight_version_key {a}→{b}, explicit dated operator directive\n"
    "# 2026-09-16 14:31 UTC): teacher-relative thought cap. Per turn the miner\n"
    "# may think up to cap_T = max(max_thought_tokens, floor(thought_cap_ratio ×\n"
    "# L_T)) tokens, L_T = the longest VALID teacher reference thought on that\n"
    "# turn, counted with the teacher's tokenizer (the references are sampled\n"
    "# first; both sides read the same references, so cap_T is the same for\n"
    "# king and challenger). It only relaxes the fixed cap on the ~11% of turns\n"
    "# where the teacher itself thinks > 1,640 tokens, and there it frees 45–47%\n"
    "# of the remaining forfeits (docs/teacher-gated-cap.md: 0 decision flips,\n"
    "# max margin move +0.0003 on the 12 stored wvk-18 duels). Action cap and\n"
    "# ref_max_tokens unchanged; serving window 110k + 5,120 + 768 < 131,072.\n"
    "# Verdicts stamp duel_params.thought_cap_rule / thought_cap_ratio /\n"
    "# thought_cap_tokenizer and per-row cap_tokens + ref_thought_tokens.\n"
    "# 0 = fixed cap (pre-wvk-20 replay path).\n"
    "thought_cap_ratio = 1.25\n")

ASSERT_LINES = (
    "ref_max_tokens = 4096\n",
    "confirmation_required = true\n",
    'crown_mode = "duel"\n',
    "k_sigma = 2.0\n",
    "min_margin = 0.002\n",
)

BANNER = "# ############################################################################\n"
HISTORY = (
    "# {a}→{b} teacher-relative miner thought cap ({date}): explicit dated\n"
    "# operator directive 2026-09-16 14:31 UTC. thought_cap_ratio = 1.25: per\n"
    "# turn the miner may think up to max(2048, floor(1.25 × L_T)) tokens, L_T\n"
    "# = the longest valid teacher reference thought on the turn in teacher\n"
    "# tokens; refs are sampled before the miner, both sides share them, so\n"
    "# the cap is identical for king and challenger. Relaxes the fixed cap only\n"
    "# where the teacher itself thinks > 1,640 tokens (~11% of turns) and there\n"
    "# frees 45–47% of the remaining forfeits; 0 decision flips, max margin\n"
    "# move +0.0003 on the stored wvk-18 duels. Action cap 768, ref cap 4096,\n"
    "# min(R,G) math, the crown bar and the confirmation slice unchanged; wvk\n"
    "# ≤ 19 verdicts replay under their stamped fixed cap. Forward-only; reign\n"
    "# 13 stands; min_submission_block unchanged.\n")


def render(src: str, date: str, wvk_to: int) -> str:
    a, b = wvk_to - 1, wvk_to
    fmt = dict(date=date, a=a, b=b)
    for line in ASSERT_LINES:
        if src.count(line) != 1:
            raise SystemExit(f"anchor changed, refusing: expected exactly one {line.strip()!r}")
    if src.count(ANCHOR_OLD) != 1:
        raise SystemExit("anchor missing: max_thought_tokens = 2048 / max_action_tokens = 768")
    if "thought_cap_ratio" in src:
        raise SystemExit("thought_cap_ratio already present in the toml")
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
    print(f"applied wvk {args.wvk_to - 1} -> {args.wvk_to}: thought_cap_ratio = 1.25 "
          f"({date}){mirrored}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
