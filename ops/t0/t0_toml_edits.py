"""The wvk 11 T0 contract edit, as code instead of a context patch.

Stage 1 (ops/t0/stage1_go_live.sh) already moved corpus_base_url to
data.affine.io and rewrote the comments around it, so a patch cut against the
pre-stage-1 tree no longer applies. This renders the T0 toml from whatever
tree it is given and is what t0_cutover.sh runs; the same function also
regenerates ops/t0/wvk11_T0.patch as a human-readable preview.

    python ops/t0/t0_toml_edits.py --preview [--toml PATH] [--date T0_DATE]
    python ops/t0/t0_toml_edits.py --apply 2026-09-09 [--toml PATH]

--apply refuses unless the file is at wvk 10 with corpus_base_url already on
data.affine.io (stage 1 done) and manifest_key still turns/manifest.json.
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TOML = REPO / "affine" / "affine.toml"
PATCH = REPO / "ops" / "t0" / "wvk11_T0.patch"

HISTORY_OLD = "# 0.001→0.002 revert (2026-08-22); 9→10 min(R,G) v5 (2026-08-27).\n"
HISTORY_NEW = (
    "# 0.001→0.002 revert (2026-08-22); 9→10 min(R,G) v5 (2026-08-27); 10→11\n"
    "# action dialects + trace-first corpus ({t0}, notice 2026-09-02): D admits\n"
    "# boxed + tool_call turns next to bash and is served as a schema_version 3\n"
    "# view over rollout traces (corpus/manifest.json on data.affine.io); score\n"
    "# unchanged; forward-only, reign stands.\n")

MANIFEST_OLD = 'manifest_key = "turns/manifest.json"\n'
MANIFEST_NEW = (
    "# {t0} (weight_version_key 10→11): the schema-3 view duel_turns@v4. Every\n"
    "# pre-T0 verdict's slice.manifest_sha256 still resolves at\n"
    "# data.affine.io/turns/manifests/{{sha}} (schema-2 history, immutable).\n"
    'manifest_key = "corpus/manifest.json"\n')

AT_T0_OLD = (
    "# At T0 manifest_key moves to corpus/manifest.json (schema_version 3); the\n"
    "# Hippius keys stay resolvable under data.affine.io/turns/** for replay.\n")
AT_T0_NEW = (
    "# Rollback (data only): manifest_key back to turns/manifest.json restores\n"
    "# the schema-2 slice population; weight_version_key does not roll back.\n")

KINDS_OLD = 'allowed_action_kinds = ["bash"]\n'
KINDS_NEW = (
    "# {t0} (weight_version_key 10→11, explicit dated operator directive):\n"
    "# boxed (math, affine-math-v1) and tool_call (wiki search, affine-wiki-v1)\n"
    "# admitted; target slice shares math 0.10 / tool_use 0.10 via strata count.\n"
    "# Notice posted 2026-09-02 (llms.txt \"Upcoming fork\", Discord, dashboard).\n"
    'allowed_action_kinds = ["bash", "tool_call", "boxed"]\n')


def render(src: str, t0: str) -> str:
    """Return the T0 contract text for a stage-1 tree; raise if the anchors
    are not exactly where they must be (never guess on the contract)."""
    checks = {
        "weight_version_key = 10": "weight_version_key = 10\n" in src,
        "corpus_base_url on data.affine.io":
            'corpus_base_url = "https://data.affine.io"\n' in src,
        "manifest_key anchor": src.count(MANIFEST_OLD) == 1,
        "history anchor": src.count(HISTORY_OLD) == 1,
        "at-T0 comment anchor": src.count(AT_T0_OLD) == 1,
        "allowed_action_kinds anchor": src.count(KINDS_OLD) == 1,
    }
    missing = [k for k, ok in checks.items() if not ok]
    if missing:
        raise SystemExit(f"affine.toml is not a stage-1 wvk-10 tree; anchors missing: {missing}")
    out = src.replace(HISTORY_OLD, HISTORY_NEW.format(t0=t0), 1)
    out = re.sub(r"^weight_version_key = 10$", "weight_version_key = 11",
                 out, count=1, flags=re.M)
    out = out.replace(MANIFEST_OLD, MANIFEST_NEW.format(t0=t0), 1)
    out = out.replace(AT_T0_OLD, AT_T0_NEW, 1)
    out = out.replace(KINDS_OLD, KINDS_NEW.format(t0=t0), 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--preview", action="store_true",
                   help="write ops/t0/wvk11_T0.patch (diff only; toml untouched)")
    g.add_argument("--apply", metavar="T0_DATE",
                   help="rewrite affine.toml in place (t0_cutover.sh only)")
    ap.add_argument("--toml", type=Path, default=TOML)
    ap.add_argument("--date", default="T0_DATE", help="date stamp for --preview")
    args = ap.parse_args()
    src = args.toml.read_text()
    if args.apply:
        if not re.fullmatch(r"20\d\d-\d\d-\d\d", args.apply):
            raise SystemExit("--apply needs the dated directive YYYY-MM-DD")
        args.toml.write_text(render(src, args.apply))
        print(f"{args.toml}: wvk 11, corpus/manifest.json, three dialects ({args.apply})")
        return 0
    new = render(src, args.date)
    diff = difflib.unified_diff(src.splitlines(keepends=True), new.splitlines(keepends=True),
                                fromfile="a/affine/affine.toml", tofile="b/affine/affine.toml")
    PATCH.write_text("".join(diff))
    print(f"wrote {PATCH} ({sum(1 for l in new.splitlines() if l) - sum(1 for l in src.splitlines() if l):+d} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
