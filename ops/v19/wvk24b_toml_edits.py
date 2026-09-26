"""wvk 24 addendum (same fork; Jacob 2026-09-23 20:47 UTC "do it"): [duel.sd_meter]
ref_min_content = 10, typ_min_refs = 2 — empty-thought reference rule. Items (3)
(k-matched control telemetry) needs no toml. --revert removes the two lines.

    python ops/v19/wvk24b_toml_edits.py --apply 2026-09-23 | --revert DATE | --preview
"""
from __future__ import annotations
import argparse, difflib, re, shutil, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
TOML = REPO / "affine" / "affine.toml"; MIRROR = REPO / "affine" / "website" / "code" / "affine.toml"
PATCH = REPO / "ops" / "v19" / "wvk24b_empty_ref_rule.patch"
ANCHOR = 'content_prefix = "refs_max"\n'
NEW = ('content_prefix = "refs_max"\n'
       "# {date} (wvk 24 addendum, explicit dated operator directive 2026-09-23 20:47 UTC):\n"
       "# a reference thought with fewer than ref_min_content content tokens does not\n"
       "# anchor typicality (it is left out of μ_c / σ_c and of the teacher control\n"
       "# instead of entering with a mean over a handful of tokens); a turn with fewer\n"
       "# than typ_min_refs content-bearing references scores min(z_R, z_A) — the\n"
       "# typicality leg is dropped for that turn. Live data: 7.5 % of references,\n"
       "# ~4.8 % of turns. 0 = every reference anchors (the wvk 22/23 rule).\n"
       "ref_min_content = 10\n"
       "typ_min_refs = 2\n")
BANNER = "# ############################################################################\n"
HIST = ("# 24 addendum ({date}): explicit dated operator directive 2026-09-23 20:47 UTC\n"
        "# (\"do it\") — same fork. (3) the teacher-vs-king control is published k-matched\n"
        "# (king scored against the same k−1 references as the held-out teacher\n"
        "# reference) and floor-dropped (king forfeits / content-floor turns excluded),\n"
        "# overall and per leg, next to the legacy construction; it is the rollback\n"
        "# signal (sign flip vs pre-fork: all −, R −, A +, Gc mixed). (6) references\n"
        "# with < 10 content tokens do not anchor typicality; < 2 content-bearing\n"
        "# references → min(z_R, z_A). Counterfactual on the last 30 verdicts (floor\n"
        "# −6 + this rule vs live): 0 flips; the rule touches 7.5 % of references and\n"
        "# drops typicality on ~4.8 % of turns. wvk 24 flipped in TWO steps at\n"
        "# consecutive duel boundaries: the floor at 20:45 UTC (chal-00678, dispatched\n"
        "# 20:46, was judged with the floor only), the empty-ref rule + k-matched\n"
        "# control from the next boundary (chal-00679 onward) — the addendum directive\n"
        "# (20:47) arrived two minutes after the floor flip had landed, and a rule\n"
        "# change is applied only at a duel boundary. duel_params.sd_meter\n"
        "# (ref_min_content / typ_min_refs) distinguishes the two.\n")
def _check(s, l):
    if s.count(l) != 1: raise SystemExit(f"anchor {l.strip()!r}: {s.count(l)} occurrences")
def _hist(out, para):
    i = out.find(BANNER); j = out.find(BANNER, i + 1)
    if i < 0 or j < 0: raise SystemExit("banner")
    head, tail = out[:i], out[i:]
    if not head.endswith("#\n"): raise SystemExit("text above banner")
    return head[:-2] + para + "#\n" + tail
def render(s, date):
    _check(s, ANCHOR); _check(s, "weight_version_key = 24\n"); _check(s, "forfeit_sd = -6\n")
    if "ref_min_content" in s: raise SystemExit("already present")
    return _hist(s.replace(ANCHOR, NEW.format(date=date)), HIST.format(date=date))
def revert(s, date):
    _check(s, "ref_min_content = 10\n"); _check(s, "typ_min_refs = 2\n")
    out = re.sub(r"# \d{4}-\d{2}-\d{2} \(wvk 24 addendum.*?typ_min_refs = 2\n", "", s, flags=re.S)
    if "ref_min_content" in out: raise SystemExit("revert failed")
    return _hist(out, f"# 24 addendum REVERTED ({date}): ref_min_content / typ_min_refs removed (operator-directed).\n")
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--toml", type=Path, default=TOML)
    g = ap.add_mutually_exclusive_group(required=True); g.add_argument("--preview", action="store_true"); g.add_argument("--apply"); g.add_argument("--revert")
    ap.add_argument("--no-mirror", action="store_true"); a = ap.parse_args()
    src = a.toml.read_text(); date = a.apply or a.revert or "YYYY-MM-DD"
    out = revert(src, date) if a.revert else render(src, date)
    if a.preview:
        PATCH.write_text("".join(difflib.unified_diff(src.splitlines(True), out.splitlines(True), "a/affine/affine.toml", "b/affine/affine.toml"))); print("wrote", PATCH); return 0
    a.toml.write_text(out)
    if not a.no_mirror and MIRROR.exists() and a.toml.resolve() == TOML.resolve(): shutil.copyfile(a.toml, MIRROR)
    print("applied", "revert" if a.revert else "addendum", date, file=sys.stderr); return 0
if __name__ == "__main__":
    raise SystemExit(main())
