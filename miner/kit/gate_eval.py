#!/usr/bin/env python3
"""Run the prior-v1 promotion gate locally: head vs your trained candidate, with the live gate's math (validator/gate/baseline_v1.py).

  python3 gate_eval.py --head ./head --head-url http://127.0.0.1:8000 --cand-url http://127.0.0.1:8001 --out gate.json \
      [--prior my_prior.jsonl] [--n 500] [--seed HEX]

What the live gate does each round (one round per head; settings in validator/gate/gate-config.public.json heldout.prior_v1):
  1. pool = miner/data/gate-pool-v1.jsonl (affine_math + dapo, 19,305 tasks).
  2. prior: per task s/n = the head's solved/sampled counts from the validator's prior sampler (method public in priors/, values
     PRIVATE). Here: --prior FILE (jsonl lines {"id", "s", "n"}, e.g. from your own head samples) or, by default, no prior
     (every task gets its env average -> the draw is close to uniform; the estimate stays unbiased either way).
  3. pi_i proportional to sqrt(E[p(1-p)]) under Beta(s+1, n-s+1), 10 % uniform floor, cap 0.10, expected size N
     (baseline_v1.inclusion); Poisson draw u_i < pi_i from the seed (baseline_v1.draw). Live seed: commit-reveal (secret + a
     Finney block hash after the upload deadline), so the real draw cannot be known in advance; here: YOUR random seed.
  4. head: 3 blocks x 4 rollouts on every drawn task (plan seed baseline_v1.plan_seed(seed, b)); candidate: k_cand = 8 rollouts
     per drawn task with an independent plan seed (baseline_v1.cand_seed(seed, epoch)).
  5. baseline_v1.estimate: Delta = sum_env share_env x Hajek mean_env(cand - head) (weights 1/pi);
     SE = sqrt(sampling + rollout variance + sigma_run^2), sigma_run = 0.003; GREEN iff Delta > 0, z > 1.00,
     affine_math cap-hit < 0.20, >= 100 tasks scored (>= 95 % of the draw).
Prompt = [system: MANDATE, user: problem], enable_thinking=True; sampling temperature 1.0, top_p 0.95, top_k 20, 12,288 output tokens (the gate's task cap from 2026-11-02; 8,192 before), context 16,384;
grading = the probe's grader (kit/grade.py), the same for both envs. Request seed = int(sha256('<plan seed>:affine_gatepool:<probe
index>:j')[:12], 16), probe index = math index, or 7496 + j for dapo:j.
Your local GPUs/vLLM build are not the gate's (vLLM 0.32.0 on 8 x B200), so expect differences of a few tenths of a point.
The old ivw-v2 procedure is in gate_eval_ivw_v2.py.
"""
import argparse, asyncio, hashlib, json, secrets, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import kitpath, sampler, grade, prompts  # noqa: E402
import baseline_v1 as BV  # noqa: E402  (verbatim live gate module, validator/gate/, via kitpath)
ENV = 'affine_gatepool'
GATE_POOL = kitpath.DATA / 'gate-pool-v1.jsonl'
GATE_POOL_SHA256 = '9e05d47c4672c8e066a15079577334f0432898ea06afc7045c82862345465a8e'


GATE_MAX_TOKENS = 12288   # the gate's per-task generation cap (Jacob 2026-11-02; 8192 before): serve with --max-model-len 16384


def gate_pool():
    raw = GATE_POOL.read_bytes()
    if hashlib.sha256(raw).hexdigest() != GATE_POOL_SHA256: raise SystemExit('gate pool sha256 mismatch')
    return {r['id']: r for r in map(json.loads, raw.decode().splitlines())}


def cells(res, tasks, k, pool):
    out = {}
    for t in tasks:
        pt = dict(n=0, c=0, w=0, u=0, b=0, x=0, tok=0, L=[])
        for j in range(k):
            o, stop = res[(t, j)]
            pt['tok'] += len(o); pt['L'].append(len(o)); pt['b'] += stop == 'budget'
            kind, _ = grade.grade(o, stop, pool[t]['answer'], pool[t]['problem'])
            pt['n'] += 1; pt[{'positive': 'c', 'negative': 'w', 'neutral': 'u'}.get(kind, 'x')] += 1
        out[t] = pt
    return out


def run(urls, model, P, tasks, k, ids, pool, conc, cfg):
    jobs = [((t, j), ids[t], sampler.request_seed(P, ENV, BV.probe_index(t, cfg), j)) for t in tasks for j in range(k)]
    return cells(asyncio.run(sampler.sample_all(urls, jobs, model=model, concurrency=conc, max_tokens=GATE_MAX_TOKENS)), tasks, k, pool)


def merge(blocks):
    out = {}
    for b in blocks:
        for t, c in b.items():
            m = out.setdefault(t, dict(n=0, c=0, w=0, u=0, b=0, x=0, tok=0, L=[]))
            for key in ('n', 'c', 'w', 'u', 'b', 'x', 'tok'): m[key] += c[key]
            m['L'] += c['L']
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--head', required=True, help='head dir (tokenizer)')
    ap.add_argument('--head-url', action='append', required=True); ap.add_argument('--cand-url', action='append', required=True)
    ap.add_argument('--model', default='m'); ap.add_argument('--n', type=int, default=None, help='expected draw size; default: live N')
    ap.add_argument('--prior', help='jsonl {"id", "s", "n"} per task (your own estimate of the head); default: none')
    ap.add_argument('--seed', default=None); ap.add_argument('--baseline', help='reuse a saved head baseline json from an earlier run')
    ap.add_argument('--out', required=True); ap.add_argument('--concurrency', type=int, default=512)
    a = ap.parse_args()
    G = json.loads((kitpath.VAL / 'gate' / 'gate-config.public.json').read_text())
    cfg = dict(BV.DEFAULTS, **((G.get('heldout') or {}).get('prior_v1') or {}))
    if a.n: cfg['N'] = a.n
    grade.init(a.head); tok = grade._G['tok']; pool = gate_pool()
    pr = {}
    if a.prior:
        for l in Path(a.prior).read_text().splitlines():
            if l.strip(): d = json.loads(l); pr[d['id']] = d
    rows = [dict(task_id=t, env=r['env'], s=float(pr.get(t, {}).get('s', 0)), n=float(pr.get(t, {}).get('n', 0)), b=0.0) for t, r in pool.items()]
    pi = BV.inclusion(rows, N=int(cfg['N']), floor=float(cfg['floor']), cap=float(cfg['cap']), shares=cfg['shares'], n_init=float(cfg['n_init']))
    if a.baseline:
        B = json.loads(Path(a.baseline).read_text()); seed = B['seed']
    else:
        seed = a.seed or secrets.token_hex(32); B = None
    S = BV.draw(pi, seed)
    ids = {t: prompts.gate_ids(tok, pool[t]['problem']) for t in S}
    nb = int(cfg['head_blocks']); kb = int(cfg['head_rollouts']) // nb
    if B is None:
        blocks = [run(a.head_url, a.model, BV.plan_seed(seed, b), S, kb, ids, pool, a.concurrency, cfg) for b in range(nb)]
        B = dict(seed=seed, blocks=blocks); Path(a.out + '.baseline.json').write_text(json.dumps(B))
    head = merge(B['blocks'])
    cand = run(a.cand_url, a.model, BV.cand_seed(seed, 0), S, int(cfg['k_cand']), ids, pool, a.concurrency, cfg)
    env_size = {}
    for r in pool.values(): env_size[r['env']] = env_size.get(r['env'], 0) + 1
    dec = BV.estimate({t: dict(env=pool[t]['env'], pi=pi[t]) for t in S}, head, cand, env_size, shares=cfg['shares'],
                      tau=float(cfg['tau_run_separate']), sigma_run_floor=float(cfg['sigma_run_floor']), bar_z=float(cfg['bar_z']),
                      z_inclusive=bool(cfg['z_inclusive']), cap_max=float(cfg['cap_max']), cap_envs=tuple(cfg['cap_envs']),
                      min_scored=int(cfg['min_scored']), min_cover=float(cfg['min_cover']))
    dec['seed'] = seed; dec['drawn'] = len(S); dec['prior'] = a.prior or 'none (env averages)'
    Path(a.out).write_text(json.dumps(dict(decision=dec, cand=cand), indent=1))
    ev = dec['evaluated']
    print(json.dumps(dict(decision=dec['decision'], reasons=dec['reasons'], gain=ev['gain'], z=ev['z'], se=ev['se'], n=ev['n'],
                          h_pass=ev['h_pass'], c_pass=ev['c_pass'], cap_rate=ev['c_cap_rate'], per_env=ev['per_env'])))


if __name__ == '__main__':
    main()
