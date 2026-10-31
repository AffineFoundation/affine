#!/usr/bin/env python3
"""Run the held-out promotion gate locally: head vs your trained candidate, with the live gate's code and math.

  python3 gate_eval.py --head ./head --head-url http://127.0.0.1:8000 --cand-url http://127.0.0.1:8001 --out gate.json [--n 2000]

Same procedure as the live gate (validator/gate/heldout.py, scoring ivw-v2; gate-config.public.json):
  1. draw n task ids from the 7,496-task pool with gate_core.draw_sample(seed, 7496, n) - here YOUR local random seed; the live
     round seed is os.urandom(32), private until the round ends, so you can never know the real sample in advance;
  2. block 0 = selection: head, k=4 per task on all n -> kept = tasks with 0 < c < 4, weight w = 1/(q(1-q)), q = (c+1)/(4+2);
  3. blocks 1..3 = head p_i runs, k=4 each, on the kept tasks only;
  4. candidate: k_cand per kept task (live: 8 since the variance-levers switch of 2026-10-31, was 4); samples j < 4 use the SAME
     request seeds as block 1 (common random numbers);
  5. heldout.score_ivw(...): gain = sum w d / sum w, SE = sqrt(se_task^2 + sigma_run^2 (1 + 1/3)); sigma_run: live policy 'pooled' =
     the value the gate froze for the current round (heldout.pooled_sigma_run over the last 20 finished rounds; gate-config
     heldout.sigma_run_pooled, also in https://affine.io/gate/heldout/current.json sigma_run.value), old policy 'per-round' =
     max(floor, run_sigma of your 3 head blocks); GREEN iff gain > 0 and z > 1.0 and estimated whole-set cap-hit < 0.20 and >= 100 kept.
  Defaults (n, k_cand, sigma_run policy) come from validator/gate/gate-config.public.json = the live flag.
Prompt = [system: MANDATE, user: problem], enable_thinking=True; sampling temperature 1.0, top_p 0.95, top_k 20, 8192 output tokens;
grading = the probe's grader (kit/grade.py). Request seed = int(sha256('P:affine_math:task:j')[:12], 16).
Your local GPUs/vLLM build are not the gate's (vLLM 0.32.0 on 8 x B200), so expect run-to-run differences of a few pp.
"""
import argparse, asyncio, json, secrets, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import prompts, sampler, grade  # noqa: E402
import heldout, gate_core  # noqa: E402  (verbatim live gate modules, via kitpath)
ENV = 'affine_math'


def cells(res, tasks, k, pool):
    out = {}
    for t in tasks:
        pt = dict(n=0, c=0, w=0, u=0, b=0, x=0, tok=0, L=[])
        for j in range(k):
            o, stop = res[(t, j)]
            pt['tok'] += len(o); pt['L'].append(len(o)); pt['b'] += stop == 'budget'
            kind, _ = grade.grade(o, stop, pool[t]['answer'], pool[t]['prompt'])
            pt['n'] += 1; pt[{'positive': 'c', 'negative': 'w', 'neutral': 'u'}.get(kind, 'x')] += 1
        out[t] = pt
    return out


def run(urls, model, P, tasks, k, ids, pool, conc):
    jobs = [((t, j), ids[t], sampler.request_seed(P, ENV, t, j)) for t in tasks for j in range(k)]
    return cells(asyncio.run(sampler.sample_all(urls, jobs, model=model, concurrency=conc)), tasks, k, pool)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--head', required=True, help='head dir (tokenizer)')
    ap.add_argument('--head-url', action='append', required=True); ap.add_argument('--cand-url', action='append', required=True)
    ap.add_argument('--model', default='m'); ap.add_argument('--n', type=int, default=None, help='default: live heldout.n (gate-config)')
    ap.add_argument('--k-cand', type=int, default=None, help='candidate samples per kept task; default: live heldout.k_cand (gate-config)')
    ap.add_argument('--sigma-run', type=float, default=None, help='fixed sigma_run; default: live policy (pooled value from gate-config, or per-round)')
    ap.add_argument('--seed', default=None); ap.add_argument('--sigma-floor', type=float, default=None, help='default: the live floor (gate-config heldout.sigma_run)')
    ap.add_argument('--baseline', help='reuse a saved baseline (blocks) json from an earlier run of the same head')
    ap.add_argument('--exclude', help='rollouts.jsonl or batch.json whose prompts to drop from the sample (off by default: the live gate has batch_tasks=0)')
    ap.add_argument('--out', required=True); ap.add_argument('--concurrency', type=int, default=512)
    a = ap.parse_args()
    grade.init(a.head); tok = grade._G['tok']; pool = prompts.pool()
    G = json.loads((Path(__file__).resolve().parent.parent / 'validator' / 'gate' / 'gate-config.public.json').read_text())
    cfg = dict(G['heldout'], cap_max=G['gate']['cap_max'])
    n = a.n or int(cfg.get('n', 1000)); kc = a.k_cand or int(cfg.get('k_cand', 4))
    if a.baseline:
        B = json.loads(Path(a.baseline).read_text()); seed = B['seed']
    else:
        seed = a.seed or secrets.token_hex(32); B = None
    S = gate_core.draw_sample(seed, len(pool), n)
    ids = {t: prompts.gate_ids(tok, pool[t]['prompt']) for t in S}
    if B is None:
        sel = run(a.head_url, a.model, heldout.baseline_plan_seed(seed, 0), S, 4, ids, pool, a.concurrency)
        kept = heldout.kept_tasks(sel)
        blocks = {b: run(a.head_url, a.model, heldout.baseline_plan_seed(seed, b), kept, 4, ids, pool, a.concurrency) for b in (1, 2, 3)}
        B = dict(seed=seed, sel={str(t): v for t, v in sel.items()}, blocks={str(b): {str(t): v for t, v in c.items()} for b, c in blocks.items()})
        Path(a.out + '.baseline.json').write_text(json.dumps(B))
    sel = {int(t): v for t, v in B['sel'].items()}; blocks = {int(b): {int(t): v for t, v in c.items()} for b, c in B['blocks'].items()}
    kept = heldout.kept_tasks(sel)
    batch = []
    if a.exclude:
        want = set()
        txt = Path(a.exclude).read_text()
        try:
            want = {g['prompt'] for g in json.loads(txt)['groups']}
        except Exception:
            want = {json.loads(l)['prompt'] for l in txt.splitlines() if l.strip()}
        batch = [t for t in kept if prompts.batch_prompt(pool[t]['prompt']) in want or pool[t]['prompt'] in want]
    cand = run(a.cand_url, a.model, heldout.cand_plan_seed(seed), kept, kc, ids, pool, a.concurrency)
    base = heldout.merged_baseline(blocks, kept)
    rs = heldout.run_sigma(sel, blocks, kept)
    pooled = (cfg.get('sigma_run_pooled') or {}).get('value') if cfg.get('sigma_run_policy') == 'pooled' else None
    if a.sigma_run is not None: sigma, pol = a.sigma_run, 'fixed (--sigma-run)'
    elif pooled is not None and a.sigma_floor is None: sigma, pol = float(pooled), 'pooled (live round value)'
    else: sigma, pol = max(cfg['sigma_run'] if a.sigma_floor is None else a.sigma_floor, rs or 0.0), 'per-round max(floor, run_sigma)'
    dec = heldout.score_ivw(sel, base, cand, kept, batch, env=ENV, bar_z=cfg['bar_z'], z_inclusive=cfg['z_inclusive'], cap_max=cfg['cap_max'],
                            min_kept=cfg['min_kept'], set_size=len(S), sigma_run=sigma, n_p_blocks=3)
    dec['run_sigma_estimate'] = rs; dec['seed'] = seed; dec['k_cand'] = kc; dec['n'] = n; dec['sigma_run_policy'] = pol
    Path(a.out).write_text(json.dumps(dict(decision=dec, cand={str(t): v for t, v in cand.items()}), indent=1))
    ev = dec['evaluated']
    print(json.dumps(dict(decision=dec['decision'], reasons=dec['reasons'], gain=ev['gain'], z=ev['z'], se=ev['se'], kept=ev['kept'],
                          h_pass=ev['h_pass'], c_pass=ev['c_pass'], cap_rate=ev['c_cap_rate'])))


if __name__ == '__main__':
    main()
