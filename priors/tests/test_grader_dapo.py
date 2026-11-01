"""Grader unit test on DAPO golds (needs the private pool: set PRIORS_DAPO_POOL; otherwise a built-in sample is used)."""
import json, os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from priors.grader import grade_text

SAMPLE = ["34", "0", "-7", "2025", "100000", "17"]


def golds():
    p = os.environ.get("PRIORS_DAPO_POOL")
    if not p: return SAMPLE
    rows = [json.loads(l) for l in open(p)]
    n = int(os.environ.get("PRIORS_GRADER_N", "500"))
    random.Random(0).shuffle(rows)
    return [r["answer"] for r in rows[:n]]


def reply(box):
    return "<think>\nwork\n</think>\n\nSo the answer is $\\boxed{%s}$" % box


def test_correct_and_wrong():
    bad = []
    for g in golds():
        if grade_text(g, reply(g), "")[0] != "positive": bad.append(("pos", g))
        w = str(int(g) + 1)
        if grade_text(g, reply(w), "")[0] != "negative": bad.append(("neg", g))
    assert not bad, bad[:20]


def test_formats():
    assert grade_text("34", "<think>a</think>\n\\boxed{34}", "")[0] == "positive"
    assert grade_text("34", "<think>a</think>\n\\[\\boxed{34}\\]", "")[0] == "positive"
    assert grade_text("34", "<think>a</think>\n\\boxed{34.0}", "")[0] == "positive"
    assert grade_text("34", "<think>a</think>\n\\boxed{\\text{34}}", "")[0] == "positive"
    assert grade_text("34", "no think block \\boxed{34}", "")[0] == "positive"           # lenient P2
    assert grade_text("34", "<think>a</think>\n\\boxed{34}", "", capped=True)[0] == "neutral"
    assert grade_text("34", "<think>a</think>\nno box", "")[0] == "neutral"
    assert grade_text("34", "<think>a</think>\n\\boxed{35}", "")[0] == "negative"
