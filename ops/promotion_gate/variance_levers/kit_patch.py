import sys
K = sys.argv[1]
p=K+"/gate_eval.py"; s=open(p).read()
R=[
("  python3 gate_eval.py --head ./head --head-url http://127.0.0.1:8000 --cand-url http://127.0.0.1:8001 --out gate.json [--n 1000]\n",
 "  python3 gate_eval.py --head ./head --head-url http://127.0.0.1:8000 --cand-url http://127.0.0.1:8001 --out gate.json [--n 2000]\n"),
("""  4. candidate: k=4 per kept task with the SAME request seeds as block 1 (common random numbers);
  5. heldout.score_ivw(...): gain = sum w d / sum w, SE = sqrt(se_task^2 + sigma_run^2 (1 + 1/3)), sigma_run = max(floor, run_sigma)
     (floor 0.0 since e354), GREEN iff gain > 0 and z > 1.0 and estimated whole-set cap-hit < 0.20 and >= 100 kept tasks.
""","""  4. candidate: k_cand per kept task (live: 8 since the variance-levers switch of 2026-10-31, was 4); samples j < 4 use the SAME
     request seeds as block 1 (common random numbers);
  5. heldout.score_ivw(...): gain = sum w d / sum w, SE = sqrt(se_task^2 + sigma_run^2 (1 + 1/3)); sigma_run: live policy 'pooled' =
     the value the gate froze for the current round (heldout.pooled_sigma_run over the last 20 finished rounds; gate-config
     heldout.sigma_run_pooled, also in https://affine.io/gate/heldout/current.json sigma_run.value), old policy 'per-round' =
     max(floor, run_sigma of your 3 head blocks); GREEN iff gain > 0 and z > 1.0 and estimated whole-set cap-hit < 0.20 and >= 100 kept.
  Defaults (n, k_cand, sigma_run policy) come from validator/gate/gate-config.public.json = the live flag.
"""),
("    ap.add_argument('--model', default='m'); ap.add_argument('--n', type=int, default=1000)\n",
 "    ap.add_argument('--model', default='m'); ap.add_argument('--n', type=int, default=None, help='default: live heldout.n (gate-config)')\n"
 "    ap.add_argument('--k-cand', type=int, default=None, help='candidate samples per kept task; default: live heldout.k_cand (gate-config)')\n"
 "    ap.add_argument('--sigma-run', type=float, default=None, help='fixed sigma_run; default: live policy (pooled value from gate-config, or per-round)')\n"),
("    cfg = dict(G['heldout'], cap_max=G['gate']['cap_max'])\n",
 "    cfg = dict(G['heldout'], cap_max=G['gate']['cap_max'])\n    n = a.n or int(cfg.get('n', 1000)); kc = a.k_cand or int(cfg.get('k_cand', 4))\n"),
("    S = gate_core.draw_sample(seed, len(pool), a.n)\n", "    S = gate_core.draw_sample(seed, len(pool), n)\n"),
("    cand = run(a.cand_url, a.model, heldout.cand_plan_seed(seed), kept, 4, ids, pool, a.concurrency)\n",
 "    cand = run(a.cand_url, a.model, heldout.cand_plan_seed(seed), kept, kc, ids, pool, a.concurrency)\n"),
("    sigma = max(cfg['sigma_run'] if a.sigma_floor is None else a.sigma_floor, rs or 0.0)\n",
 "    pooled = (cfg.get('sigma_run_pooled') or {}).get('value') if cfg.get('sigma_run_policy') == 'pooled' else None\n"
 "    if a.sigma_run is not None: sigma, pol = a.sigma_run, 'fixed (--sigma-run)'\n"
 "    elif pooled is not None and a.sigma_floor is None: sigma, pol = float(pooled), 'pooled (live round value)'\n"
 "    else: sigma, pol = max(cfg['sigma_run'] if a.sigma_floor is None else a.sigma_floor, rs or 0.0), 'per-round max(floor, run_sigma)'\n"),
("    dec['run_sigma_estimate'] = rs; dec['seed'] = seed\n", "    dec['run_sigma_estimate'] = rs; dec['seed'] = seed; dec['k_cand'] = kc; dec['n'] = n; dec['sigma_run_policy'] = pol\n"),
]
for a,b in R:
    assert s.count(a)==1, a[:60]; s=s.replace(a,b)
open(p,"w").write(s)
p=K+"/simulate_turn.sh"; s=open(p).read()
for a,b in [("N_GATE (1000)","N_GATE (live heldout.n, 2000)"),('N_GATE="${N_GATE:-1000}"','N_GATE="${N_GATE:-2000}"')]:
    assert s.count(a)==1, a; s=s.replace(a,b)
open(p,"w").write(s)
print("ok")
