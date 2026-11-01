"""Grader for the prior-table sampler = the live affine_math grade path of the probe worker:
native final-channel projection + math_verify equality (Allan's emulation of verify.py), then Allan's lenient fixes
(catoneone/sn120-k1-fixes 786cdb9, VERIFIER_FIX 'allan-lenient-v1-786cdb9'). allan_src.py is byte-identical to the live
probe_worker._ALLAN_SRC (sha256 pinned below); it is exec'd with the same two-line import prefix the probe worker uses.
A capped reply (finish_reason 'length', no EOS) is never correct (live budget skip)."""
import hashlib, os
ALLAN_SHA256 = "6b58684ca85a3f750de793e4ba1aeca5a45d8b44fb8c00b403cf5e7372823057"
VERIFIER_FIX = "allan-lenient-v1-786cdb9"
_NS = {}


def ns():
    if not _NS:
        src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "allan_src.py")).read()
        if hashlib.sha256(src.encode()).hexdigest() != ALLAN_SHA256:
            raise RuntimeError("allan_src.py sha256 mismatch")
        g = {"__name__": "allan_lenient_v1"}
        exec(compile("import re\nfrom math_verify import parse, verify\n" + src, "allan_lenient_v1", "exec"), g)
        _NS.update(g)
    return _NS


def grade_text(gold: str, text: str, prompt: str, capped: bool = False):
    """-> (kind, fix). kind in positive | negative | neutral. text = decoded reply WITHOUT the EOS token, special tokens kept
    (so '<think>' / '</think>' are visible exactly as the probe worker decodes them)."""
    if capped:
        return "neutral", "budget"
    g = ns()
    box, viol, pfix = g["lenient_project"](text)
    if box is None:
        return "neutral", viol or "no-box"
    ok, efix = g["lenient_equal"](str(gold), box, prompt)
    if ok:
        return "positive", ("+".join(x for x in (pfix, efix) if x) or "live")
    return "negative", None
