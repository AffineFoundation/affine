"""Stage the v7 fork bundle in affine.toml: think-close forfeit + `text` dialect.

Built 2026-09-08 on the operator's order after the SWE-bench Pro / Claude
Code post-mortem ("bundle into the next fork: require_think_close = true,
admit the text dialect fed by trace final replies ... one wvk event, one
notice"). The FLIP is a weight_version_key event and only runs with an
explicit, dated directive that names the key (AGENTS.md / .cursor/rules).
Until then this script only previews.

    python ops/v7/v7_toml_edits.py --preview [--toml PATH]     # writes ops/v7/wvk13_T0.patch
    python ops/v7/v7_toml_edits.py --apply DATE [--wvk-to N] [--toml PATH]

--apply refuses unless require_think_close is false, `text` is not yet in
allowed_action_kinds and weight_version_key == N-1.

What flips (the why lives next to each knob in affine.toml):
  require_think_close   false -> true      a rollout that never closes </think>
                                           forfeits (forfeit_turn_score)
  allowed_action_kinds  + "text"           the visible final reply of a rollout
                                           that ended by itself is a scored
                                           action (affine/dialects.py `text`)
  weight_version_key    N-1 -> N           (+ one history paragraph)

After --apply the operator runbook is the wvk-11 one (ops/t0/t0_cutover.sh
steps 2-4): commit, `pm2 restart affine-validator`, eval-pod redeploy between
duels (the pod reads its own affine.toml), one fold (`ops/corpus_build.py`)
so the next epoch admits the text turns already present in the traces,
`build_llms_txt.py` with the notice moved to history, Discord post.
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TOML = REPO / "affine" / "affine.toml"
PATCH = REPO / "ops" / "v7" / "wvk13_T0.patch"

THINK_OLD = "require_think_close = false\n"
THINK_NEW = (
    "# {date} (weight_version_key {a}→{b}, explicit dated operator directive):\n"
    "# on. A reply that never closes </think> scores forfeit_turn_score.\n"
    "require_think_close = true\n")

KINDS_OLD = 'allowed_action_kinds = ["bash", "tool_call", "boxed"]\n'
KINDS_NEW = (
    "# {date} (weight_version_key {a}→{b}, explicit dated operator directive):\n"
    "# text admitted — the visible final reply of a rollout that stopped by\n"
    "# itself (agent_completed, untruncated) is an action; the reply the\n"
    "# model chose to end on. Fed by the traces already published (bash /\n"
    "# pi / claude_code harness reports, wiki answers): no new rollouts, the\n"
    "# next fold picks them up. Why: min(R,G) never scored the visible\n"
    "# message, so kings learned to say nothing outside <think>\n"
    "# (SWE-bench Pro / Claude Code post-mortem 2026-09-08).\n"
    'allowed_action_kinds = ["bash", "tool_call", "boxed", "text"]\n')

BANNER = "# ############################################################################\n"
HISTORY = (
    "# {a}→{b} think-close forfeit + text dialect ({date}): a rollout without\n"
    "# </think> forfeits (require_think_close), and the visible final reply of\n"
    "# a self-ended rollout is a scored action (allowed_action_kinds += text).\n"
    "# min(R,G) math unchanged. Forward-only; reign stands.\n")


def render(src: str, date: str, wvk_to: int) -> str:
    a, b = wvk_to - 1, wvk_to
    if src.count(THINK_OLD) != 1:
        raise SystemExit("anchor missing or already flipped: require_think_close = false")
    if src.count(KINDS_OLD) != 1:
        raise SystemExit('anchor missing or already flipped: allowed_action_kinds = ["bash", "tool_call", "boxed"]')
    key_old = f"weight_version_key = {a}\n"
    if src.count(key_old) != 1:
        raise SystemExit(f"weight_version_key is not {a}")
    fmt = dict(date=date, a=a, b=b)
    out = src.replace(THINK_OLD, THINK_NEW.format(**fmt))
    out = out.replace(KINDS_OLD, KINDS_NEW.format(**fmt))
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
                   help="write ops/v7/wvk13_T0.patch; toml untouched")
    g.add_argument("--apply", metavar="DATE",
                   help="explicit dated operator directive (YYYY-MM-DD)")
    ap.add_argument("--wvk-to", type=int, help="new weight_version_key (default: current + 1)")
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
    print(f"applied v7 bundle: weight_version_key {args.wvk_to - 1} -> {args.wvk_to}, "
          f"require_think_close = true, allowed_action_kinds += text ({date})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
