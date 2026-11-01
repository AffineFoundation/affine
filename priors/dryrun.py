"""Dry run of the prior-weighted baseline (NOT LIVE). Uses the private prior table to draw a set exactly as the live gate would,
then reports what the gate would look like vs today's held-out gate (uniform 1000 of affine_math, selection block k=4, keep 0<c<4,
3x4 head p_i blocks, k_cand=4, ivw-v2). Gate SE is simulated (Monte Carlo): true p per task ~ its Beta posterior, the head and a
null candidate (same p) are sampled with the respective rollout counts and scored by heldout.score_ivw (refine.score for the new
design). Also: power vs a candidate that is better by a logit shift. Output: private JSON report (never publish: contains task ids).
usage: dryrun.py TABLE.sqlite OUT.json [--n 1000] [--reps 200] [--blocks 4] [--k-block 4] [--k-cand 4] [--shift 0.25]"""
import json, math, random, statistics, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from priors import selection as S, refine as RF
from priors.table import PriorTable


def arg(k, d, f=float):
    return f(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d


def binom(rng, n, p):
    return sum(rng.random() < p for _ in range(n))


def shift(p, s):
    if p <= 0 or p >= 1: return p
    x = math.log(p / (1 - p)) + s; return 1 / (1 + math.exp(-x))


def main():
    T = PriorTable(sys.argv[1]); out_path = sys.argv[2]
    N = arg("--n", 1000, int); REPS = arg("--reps", 200, int); BL = arg("--blocks", 4, int); KB = arg("--k-block", 4, int)
    KC = arg("--k-cand", 4, int); SH = arg("--shift", 0.25); SIG = arg("--sigma-run", 0.022)
    rows = T.rows()      # the whole active set: unseen tasks (n = 0) carry the env prior weight, no hard filter
    cfg = dict(S.DEFAULTS, n=N)
    pri = S.env_priors(rows, cfg["prior"], cfg["kappa_min"], cfg["kappa_max"], cfg["eb_min_n"])
    seed = S.new_seed(); d = S.draw(rows, seed, cfg)
    w_all = S.weights(rows, pri); env_of = {r["task_id"]: r["env"] for r in rows}
    W_univ = sum(w_all) / len(w_all); math_rows = [(r, w) for r, w in zip(rows, w_all) if r["env"] == "affine_math"]
    W_math = sum(w for _, w in math_rows) / max(1, len(math_rows))
    mass = sum(c["w"] for c in d["cells"].values())
    rep = dict(at=time.time(), table_head=T.head(), table_digest=T.digest(), tasks_in_table=len(rows), tasks_with_data=sum(1 for r in rows if r["n"] > 0),
               by_env_in_table={e: dict(tasks=sum(1 for r in rows if r["env"] == e), with_data=sum(1 for r in rows if r["env"] == e and r["n"] > 0))
                                for e in sorted(set(env_of.values()))},
               priors=pri, draw=dict(n=d["n"], alloc=d["alloc"], env_share={e: d["alloc"][e] / d["n"] for e in d["alloc"]},
                                     commitment=S.commitment(seed, d["ids"]), seed_note="fresh os.urandom seed, dry run only (not stored)",
                                     pi_min=min(c["pi"] for c in d["cells"].values()), pi_max=max(c["pi"] for c in d["cells"].values()),
                                     certain=sum(c["pi"] >= 1 for c in d["cells"].values())),
               pvar_mass=dict(drawn=mass, uniform_same_size_over_active_set=N * W_univ, uniform_1000_affine_math_today=N * W_math,
                              ratio_vs_uniform_active=mass / (N * W_univ), ratio_vs_today_pool=mass / (N * W_math)))
    rep["gate_sim"] = {}
    for SIG_ in (0.0, SIG):
        rep["gate_sim"]["sigma_run=%g" % SIG_] = simulate(rows, pri, cfg, N, REPS, BL, KB, KC, SH, SIG_, math_rows)
    Path(out_path).write_text(json.dumps(rep, indent=1, default=str)); Path(out_path).chmod(0o600)
    print(json.dumps({k: rep[k] for k in ("draw", "pvar_mass", "gate_sim", "by_env_in_table", "priors")}, indent=1, default=str))


def simulate(rows, pri, cfg, N, REPS, BL, KB, KC, SH, SIG, math_rows):
    rng = random.Random(1); new_se, new_set, new_z0, new_z1, old_se, old_set, old_z0, old_z1, old_kept = [], [], [], [], [], [], [], [], []
    by_id = {r["task_id"]: r for r in rows}
    def true_p(r):
        p = pri[r["env"]]; return rng.betavariate(r["s"] + p["a0"], r["n"] - r["s"] + p["b0"])
    for k in range(REPS):
        dk = S.draw(rows, "%064x" % rng.getrandbits(256), cfg)
        P = {t: true_p(by_id[t]) for t in dk["ids"]}
        base = {t: dict(n=BL * KB, c=binom(rng, BL * KB, P[t]), b=0) for t in dk["ids"]}
        for s_, Z in ((0.0, new_z0), (SH, new_z1)):
            cand = {t: dict(n=KC, c=binom(rng, KC, shift(P[t], s_)), b=0) for t in dk["ids"]}
            r = RF.score(dk, base, cand, (), sigma_run=SIG, n_p_blocks=BL, min_kept=1)
            Z.append(r["evaluated"]["z"])
            if s_ == 0.0: new_se.append(r["evaluated"]["se"]); new_set.append(r["evaluated"]["se_task"])
        # today's gate
        S0 = rng.sample(math_rows, min(N, len(math_rows))); sel = {}; P0 = {}
        for i, (r, _) in enumerate(S0):
            P0[i] = true_p(r); sel[i] = dict(n=4, c=binom(rng, 4, P0[i]), b=0)
        H = RF.heldout(); kept = H.kept_tasks(sel); old_kept.append(len(kept))
        b0 = {i: dict(n=3 * 4, c=binom(rng, 12, P0[i]), b=0) for i in kept}
        for s_, Z in ((0.0, old_z0), (SH, old_z1)):
            c0 = {i: dict(n=KC, c=binom(rng, KC, shift(P0[i], s_)), b=0) for i in kept}
            r = H.score_ivw(sel, b0, c0, kept, [], env="affine_math", min_kept=1, set_size=len(sel), sigma_run=SIG, n_p_blocks=3)
            Z.append(r["evaluated"]["z"])
            if s_ == 0.0: old_se.append(r["evaluated"]["se"]); old_set.append(r["evaluated"]["se_task"])
    q = lambda xs: dict(mean=statistics.mean(xs), sd=statistics.pstdev(xs)) if xs else None
    pw = lambda zs, bar=1.0: sum(z > bar for z in zs) / len(zs) if zs else None
    return dict(reps=REPS, sigma_run=SIG, logit_shift_alt=SH,
                           new=dict(design="prior-PPS %d tasks, %dx%d head rollouts, k_cand %d, ivw prior-q weights, no filter" % (N, BL, KB, KC),
                                    se_task=q(new_set), se_total=q(new_se), z_null=q(new_z0), z_alt=q(new_z1),
                                    green_rate_null=pw(new_z0), power_alt=pw(new_z1)),
                           today=dict(design="uniform %d affine_math, sel k=4 keep 0<c<4, 3x4 p_i, k_cand %d, ivw-v2" % (N, KC),
                                      kept=q(old_kept), se_task=q(old_set), se_total=q(old_se), z_null=q(old_z0), z_alt=q(old_z1),
                                      green_rate_null=pw(old_z0), power_alt=pw(old_z1)))


if __name__ == "__main__":
    main()
