"""wvk 22 -> 23: thought cap 2048 -> 4096 (refs 4096 -> 4864) + one-sided-on-the-
long-end typicality (content_prefix = refs_max). --revert undoes it.

Explicit dated operator directive (Jacob Steeves, 2026-09-22 17:00 UTC, after
the benchsuite audit and the four options): "all of them and also 3 flipped".

What flips (--apply DATE):
  max_thought_tokens        2048 -> 4096     miner thought cap (thought_cap_ratio 1.25 unchanged)
  ref_max_tokens            4096 -> 4864     teacher reference budget (thought + action) — must be
                                             >= max_thought_tokens + max_action_tokens (config validator);
                                             4864 = 4096 + 768, so the teacher can think the full
                                             miner cap and still act. 8192 not taken: KV is not the
                                             constraint (1.1M tokens/replica), the teacher's k = 3
                                             reference samples per turn set the wall time (~2x at
                                             8192) and 1.25 x L_T would lift the miner cap to ~9.3k.
  sd_meter.content_prefix   (absent = none) -> "refs_max"   typicality on the first K content tokens
                                             of the miner's thought, K = max_i n_content(z_C^i)
  weight_version_key        22 -> 23  (+ one history paragraph)
Asserted, not edited: score_mode sd_min_rga, thought_rendering as_generated, n_turns 1000,
max_action_tokens 768, thought_cap_ratio 1.25, min_margin_sd 0.2, forfeit_sd -12, k_sigma 2.0 (x2).

    python ops/v18/wvk23_toml_edits.py --preview
    python ops/v18/wvk23_toml_edits.py --apply 2026-09-22
    python ops/v18/wvk23_toml_edits.py --revert 2026-09-22
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
PATCH = REPO / "ops" / "v18" / "wvk23_thought_cap_one_sided_g.patch"

ASSERT = ('score_mode = "sd_min_rga"\n', 'thought_rendering = "as_generated"\n', "n_turns = 1000\n",
          "max_action_tokens = 768\n", "thought_cap_ratio = 1.25\n", "min_margin_sd = 0.2\n", "forfeit_sd = -12\n")
CAP_OLD = "max_thought_tokens = 2048\n"
CAP_NEW = ("# {date} (wvk {a}→{b}, explicit dated operator directive 2026-09-22 17:00 UTC):\n"
           "# 2048 → 4096. Kings think less every generation (GPQA chains 5.4k → 3.3k →\n"
           "# 2.2k tokens over reigns 19 → 21); the cap is not binding on D (forfeits\n"
           "# 0.2 %) but it bounds the teacher-relative cap and the references, so it\n"
           "# is raised across the board. ~3x duel cost accepted.\n"
           "max_thought_tokens = 4096\n")
REF_OLD = "ref_max_tokens = 4096\n"
REF_NEW = ("# {date} (wvk {a}→{b}): 4096 → 4864 = max_thought_tokens + max_action_tokens, so\n"
           "# a reference may think the full miner cap and still act (the validator\n"
           "# requires ref_max_tokens >= thought + action). Not 8192: the k = 3\n"
           "# reference samples per turn set the wall time, KV is not the constraint.\n"
           "ref_max_tokens = 4864\n")
CP_ANCHOR = "cross_echo = true\n"
CP_NEW = ("cross_echo = true\n"
          "# {date} (wvk {a}→{b}, directive 17:00 UTC \"3 flipped\" — typicality one-sided on\n"
          "# the long end): only the first K content tokens of the miner's thought are\n"
          "# scored, K = the largest content-token count among the turn's k references.\n"
          "# A thought longer / more deliberate than the teacher's is not penalised\n"
          "# for the extra; the two-sided band still applies to the scored prefix, so\n"
          "# filler stays below and a pasted reference thought stays above (RT-11 /\n"
          "# RT-12 guards intact). Probe: ops/v18/probe_a.txt, probe_b.txt; project\n"
          "# store internal/wvk23/g-one-sided-probe.md. \"none\" = the wvk-22 rule.\n"
          "content_prefix = \"refs_max\"\n")
BANNER = "# ############################################################################\n"
HISTORY = (
    "# {a}→{b} thought cap 4096 + one-sided-long typicality ({date}): explicit dated\n"
    "# operator directive 2026-09-22 17:00 UTC (\"all of them and also 3 flipped\",\n"
    "# after the benchsuite thought-shrink audit). max_thought_tokens 2048 → 4096,\n"
    "# ref_max_tokens 4096 → 4864 (refs may think the full miner cap and act);\n"
    "# [duel.sd_meter].content_prefix = refs_max: the miner's thought is judged for\n"
    "# typicality on its first K content tokens, K = the teacher's longest reference\n"
    "# in content tokens — extra deliberation is unscored, the short / filler side\n"
    "# keeps its penalty, the two-sided band on the scored prefix keeps the pasting\n"
    "# guard. Offline probe on the last 30 wvk-22 verdicts + a 173-turn re-echo:\n"
    "# 0/30 decisions change, truncation touches 28 % of miner thoughts by +0.03…\n"
    "# +0.07 sd on average, typicality-leg teacher-vs-king control +0.36 → +0.27 sd\n"
    "# (z 3.8 → 3.0). Everything else in [duel] unchanged; forward-only, reign 21\n"
    "# stands, min_submission_block unchanged. ~3x duel cost accepted.\n")
REVERT_HISTORY = (
    "# {a}→{b} ROLLBACK to the wvk-22 settings ({date}): explicit operator-directed\n"
    "# revert (control z sign flip on the first wvk-23 verdicts). max_thought_tokens\n"
    "# 4096 → 2048, ref_max_tokens 4864 → 4096, content_prefix refs_max → none.\n"
    "# Verdicts judged under wvk 23 in between stand.\n")


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


def render(src, date, a=22, b=23):
    fmt = dict(date=date, a=a, b=b)
    for line in ASSERT + (CAP_OLD, REF_OLD, CP_ANCHOR, f"weight_version_key = {a}\n"):
        _check(src, line)
    if src.count("k_sigma = 2.0\n") != 2:
        raise SystemExit("k_sigma = 2.0 expected twice")
    if "content_prefix" in src:
        raise SystemExit("content_prefix already present")
    out = src.replace(CAP_OLD, CAP_NEW.format(**fmt)).replace(REF_OLD, REF_NEW.format(**fmt))
    out = out.replace(CP_ANCHOR, CP_NEW.format(**fmt)).replace(f"weight_version_key = {a}\n", f"weight_version_key = {b}\n")
    return _history(out, HISTORY.format(**fmt))


def render_revert(src, date):
    for line in ("max_thought_tokens = 4096\n", "ref_max_tokens = 4864\n", 'content_prefix = "refs_max"\n', "weight_version_key = 23\n"):
        _check(src, line)
    out = src.replace("max_thought_tokens = 4096\n", "max_thought_tokens = 2048\n").replace("ref_max_tokens = 4864\n", "ref_max_tokens = 4096\n")
    out = out.replace('content_prefix = "refs_max"\n', 'content_prefix = "none"\n').replace("weight_version_key = 23\n", "weight_version_key = 22\n")
    return _history(out, REVERT_HISTORY.format(date=date, a=23, b=22))


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
    what = "REVERT wvk 23 -> 22" if a.revert else "wvk 22 -> 23: max_thought_tokens 4096, ref_max_tokens 4864, content_prefix refs_max"
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
