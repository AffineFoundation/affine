"""wvk 25 scoring knobs (fork worker's part; the lead's deploy_teacher_swap.sh calls this
AFTER the teacher/window flip and the weight_version_key 24 -> 25 bump).

  --apply DATE : miner_empty_rule floor -> drop_typ, empty_gate_ratio 0.0 -> 2.0,
                 r_cap_teacher false -> true, seq_enabled false -> true (+ history paragraph)
  --revert DATE: the four knobs back (rollback; wvk integer handled by the lead's script)
  --preview    : write ops/v20/wvk25_rules.patch
Asserted, not edited: seq_look_every 100, seq_k 2.6, seq_consecutive 2, min_margin_sd 0.2,
forfeit_sd -6, ref_min_content 10, typ_min_refs 2, content_prefix refs_max.
"""
from __future__ import annotations
import argparse, difflib, re, shutil, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
TOML = REPO / "affine" / "affine.toml"; MIRROR = REPO / "affine" / "website" / "code" / "affine.toml"
PATCH = REPO / "ops" / "v20" / "wvk25_rules.patch"
ASSERT = ("seq_look_every = 100\n", "seq_k = 2.6\n", "seq_consecutive = 2\n", "min_margin_sd = 0.2\n", "forfeit_sd = -6\n",
          "ref_min_content = 10\n", "typ_min_refs = 2\n", 'content_prefix = "refs_max"\n')
FLIPS = (('miner_empty_rule = "floor"\n', 'miner_empty_rule = "drop_typ"\n'),
         ("empty_gate_ratio = 0.0\n", "empty_gate_ratio = 2.0\n"),
         ("r_cap_teacher = false\n", "r_cap_teacher = true\n"),
         ("seq_enabled = false\n", "seq_enabled = true\n"))
BANNER = "# ############################################################################\n"
HIST = ("# 25 scoring bundle ({date}, with the GLM-5.3-Flash teacher swap + 262k window;\n"
        "# explicit dated operator directive 2026-09-26 09:09 UTC \"lets do this switch\",\n"
        "# T0 2026-09-30 14:00 UTC): miner_empty_rule drop_typ (a miner thought with < 10\n"
        "# content tokens scores min(z_R, z_A) — the wvk-24 reference rule applied to the\n"
        "# miner; closes the empty-thought floor channel that paid reigns 17–20),\n"
        "# empty_gate_ratio 2.0 (a side above 2× the teacher's empty share keeps the floor\n"
        "# on those turns), r_cap_teacher (z_R capped at 0), seq_enabled (looks every 100\n"
        "# turns, crown when margin − 2.6·SE > δ on two consecutive looks, futility stop,\n"
        "# else the full slice). Counterfactual on the last 40 Qwen-ref verdicts: parity\n"
        "# 40/40 at the old knobs; bundle flips 1/40 (chal-00662's crown → +0.14 < δ, the\n"
        "# empty-thought asymmetry), paired sd −14 %, control_matched −0.24 → +0.37.\n"
        "# Forward-only, reign 21 stands, min_submission_block unchanged.\n")
REVERT_HIST = "# 25 scoring bundle REVERTED ({date}): the four knobs back to the wvk-24 values (operator-directed).\n"
def _check(s, l):
    if s.count(l) != 1: raise SystemExit(f"anchor {l.strip()!r}: {s.count(l)} occurrences")
def _hist(out, para):
    i = out.find(BANNER); j = out.find(BANNER, i + 1)
    if i < 0 or j < 0: raise SystemExit("banner")
    head, tail = out[:i], out[i:]
    if not head.endswith("#\n"): raise SystemExit("text above banner")
    return head[:-2] + para + "#\n" + tail
def render(s, date, revert=False):
    for l in ASSERT: _check(s, l)
    out = s
    for old, new in FLIPS:
        if revert: old, new = new, old
        _check(out, old); out = out.replace(old, new)
    return _hist(out, (REVERT_HIST if revert else HIST).format(date=date))
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--toml", type=Path, default=TOML)
    g = ap.add_mutually_exclusive_group(required=True); g.add_argument("--preview", action="store_true"); g.add_argument("--apply"); g.add_argument("--revert")
    ap.add_argument("--no-mirror", action="store_true"); a = ap.parse_args()
    src = a.toml.read_text(); date = a.apply or a.revert or "YYYY-MM-DD"
    if (a.apply or a.revert) and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date): raise SystemExit("date")
    out = render(src, date, revert=bool(a.revert))
    if a.preview:
        PATCH.write_text("".join(difflib.unified_diff(src.splitlines(True), out.splitlines(True), "a/affine/affine.toml", "b/affine/affine.toml"))); print("wrote", PATCH); return 0
    a.toml.write_text(out)
    if not a.no_mirror and MIRROR.exists() and a.toml.resolve() == TOML.resolve(): shutil.copyfile(a.toml, MIRROR)
    print("applied", "revert" if a.revert else "wvk 25 rules", date, file=sys.stderr); return 0
if __name__ == "__main__":
    raise SystemExit(main())
