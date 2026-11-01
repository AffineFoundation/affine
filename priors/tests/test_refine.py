import os, random, sys
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from priors import refine as RF, selection as S

pytestmark = pytest.mark.skipif(not os.path.exists(RF.HELDOUT), reason="live heldout.py not available (set PRIORS_HELDOUT_PY)")


def test_score_null_and_better():
    rng = random.Random(3)
    rows = [dict(task_id="dapo:%d" % i, env="dapo", n=8, s=rng.randint(1, 7), b=0) for i in range(800)]
    d = S.draw(rows, "11" * 32, dict(n=300, min_share={}))
    P = {t: d["cells"][t]["q"] for t in d["ids"]}
    base = {t: dict(n=16, c=sum(rng.random() < P[t] for _ in range(16)), b=0) for t in d["ids"]}
    worse = {t: dict(n=4, c=sum(rng.random() < P[t] * 0.5 for _ in range(4)), b=0) for t in d["ids"]}
    better = {t: dict(n=4, c=sum(rng.random() < min(1, P[t] * 1.5) for _ in range(4)), b=0) for t in d["ids"]}
    rw = RF.score(d, base, worse, (), sigma_run=0.0); rb = RF.score(d, base, better, (), sigma_run=0.0)
    assert rw["decision"] == "red" and rw["evaluated"]["gain"] < 0 and rw["hajek_gain"] < 0
    assert rb["decision"] == "green" and rb["evaluated"]["gain"] > 0 and rb["hajek_gain"] > 0
    assert rb["evaluated"]["n"] == 300
    # training-batch tasks are dropped from both sides
    rb2 = RF.score(d, base, better, d["ids"][:50], sigma_run=0.0)
    assert rb2["evaluated"]["n"] == 250


def test_plan_and_merge():
    p = RF.plan(dict(ids=["math:1", "dapo:2"]), "22" * 32, blocks=4, k_block=4)
    assert len(p["blocks"]) == 4 and p["cand_plan_seed"] == p["blocks"][1]["plan_seed"]
    m = RF.merge_blocks([{"math:1": dict(n=4, c=1, b=0)}, {"math:1": dict(n=4, c=3, b=1)}])
    assert m["math:1"] == dict(n=8, c=4, b=1)
