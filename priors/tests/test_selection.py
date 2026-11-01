import os, random, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from priors import selection as S
from priors.table import PriorTable


def test_posterior_weight_examples():
    assert abs(S.post_var(1, 1) - 1 / 6) < 1e-9          # 1/1  -> 0.167
    assert abs(S.post_var(16, 16) - 17 / (18 * 19)) < 1e-9  # 16/16 -> 0.050
    assert abs(S.post_var(64, 64) - 65 / (66 * 67)) < 1e-9  # 64/64 -> 0.015
    assert abs(S.post_var(0, 0) - 1 / 6) < 1e-9
    assert S.post_var(8, 16) > S.post_var(1, 1) > S.post_var(16, 16) > S.post_var(64, 64)


def rows(seed=0, n_math=600, n_dapo=900):
    rng = random.Random(seed); out = []
    for i in range(n_math):
        p = rng.betavariate(5, 0.6); n = 4; out.append(dict(task_id="math:%d" % i, env="affine_math", n=n, s=sum(rng.random() < p for _ in range(n)), b=0))
    for i in range(n_dapo):
        p = rng.betavariate(0.7, 0.9); n = rng.choice([0, 4, 8]); out.append(dict(task_id="dapo:%d" % i, env="dapo", n=n, s=sum(rng.random() < p for _ in range(n)), b=0))
    return out


def test_draw_size_shares_and_pi():
    R = rows(); d = S.draw(R, "ab" * 32, dict(n=300, min_share={"affine_math": 0.3, "dapo": 0.3}))
    assert d["n"] == 300 and len(set(d["ids"])) == 300
    n_math = sum(c["env"] == "affine_math" for c in d["cells"].values())
    assert n_math >= 90 and 300 - n_math >= 90
    assert all(0 < c["pi"] <= 1 for c in d["cells"].values())
    assert d["ids"] == S.draw(R, "ab" * 32, dict(n=300))["ids"]          # deterministic given seed
    assert d["ids"] != S.draw(R, "cd" * 32, dict(n=300))["ids"]


def test_inclusion_frequencies_match_pi():
    R = rows(1, 200, 300); cfg = dict(n=100, min_share={"affine_math": 0.3, "dapo": 0.3}, prior="uniform")
    pri = S.env_priors(R, "uniform"); w = S.weights(R, pri)
    cnt = {}; T = 1500; pis = {}
    for k in range(T):
        d = S.draw(R, "%064x" % (k + 1), cfg)
        for t, c in d["cells"].items(): cnt[t] = cnt.get(t, 0) + 1; pis[t] = c["pi"]
    err = [abs(cnt[t] / T - pis[t]) for t in pis if pis[t] > 0.2]
    assert sum(err) / len(err) < 0.03


def test_eb_prior_and_min_share():
    R = rows(); pri = S.env_priors(R, "eb")
    assert pri["affine_math"]["m"] > 0.7 and pri["dapo"]["m"] < 0.6
    assert all(S.DEFAULTS["kappa_min"] <= p["kappa"] <= S.DEFAULTS["kappa_max"] for p in pri.values())


def test_commitment():
    ids = ["math:3", "dapo:10"]
    assert S.commitment("00" * 32, ids) == S.commitment("00" * 32, list(reversed(ids)))


def test_table_discount():
    with tempfile.TemporaryDirectory() as d:
        T = PriorTable(os.path.join(d, "t.sqlite"), discount=0.5)
        T.set_head("h1"); T.ingest([dict(task_id="math:1", env="affine_math", head="h1", n=4, c=4, b=0)])
        r = T.rows()[0]; assert (r["s"], r["n"], r["n_head"]) == (4, 4, 4)
        T.set_head("h2"); r = T.rows()[0]
        assert (r["s"], r["n"], r["n_head"], r["disc_from"]) == (2, 2, 0, "h1")
        T.ingest([dict(task_id="math:1", env="affine_math", head="h2", n=4, c=0, b=1)])
        r = T.rows()[0]; assert (r["s"], r["n"], r["s_head"], r["n_head"], r["b_head"]) == (2, 6, 0, 4, 1)
        T.ingest([dict(task_id="math:1", env="affine_math", head="h1", n=4, c=4, b=0)])   # straggler from the old head: discounted
        r = T.rows()[0]; assert (r["s"], r["n"]) == (4, 8)
        assert T.set_head("h2") is False
