import json, os, sys, statistics
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'gate'))
import gate_core as G
D = json.load(open(os.path.join(os.path.dirname(__file__), '..', 'data', 'history.json')))
probes = {ck: {int(k): v for k, v in p['per_task'].items()} for ck, p in D['probes'].items()}
pmeta = D['probes']
EXPECT_RED = {128, 129, 140, 142, 144, 146, 148}
SEEDS = [("%064x" % (0xA11CE + i * 7919)) for i in range(int(sys.argv[1]) if len(sys.argv) > 1 else 20)]
rows = []
for e in range(125, 151):
    r = D['epochs'][str(e)]
    if r.get('status') != 'updated':
        rows.append(dict(epoch=e, skip=r.get('status') or 'no-final')); continue
    H, C = r['input'], r['output']
    if H not in probes or C not in probes:
        rows.append(dict(epoch=e, skip='missing probe H=%s C=%s' % (H in probes, C in probes))); continue
    h, c = probes[H], probes[C]; batch = r.get('batch') or []
    decs = [G.decide(h, c, G.draw_sample(s, 7496, 1000), batch) for s in SEEDS]
    full = G.decide(h, c, list(range(7496)), batch, min_tasks=1)
    greens = sum(d['decision'] == 'green' for d in decs)
    rows.append(dict(epoch=e, H=H[:8], C=C[:8], h_partial=pmeta[H]['partial'], h_n=pmeta[H]['n'], c_partial=pmeta[C]['partial'], c_n=pmeta[C]['n'],
                     n_eval=statistics.mean(d['evaluated']['n'] for d in decs), z_mean=statistics.mean(d['evaluated']['z'] for d in decs),
                     z_min=min(d['evaluated']['z'] for d in decs), z_max=max(d['evaluated']['z'] for d in decs),
                     cap=statistics.mean(d['evaluated']['c_cap_rate'] or 0 for d in decs),
                     green_frac=greens / len(decs), decision=('green' if greens * 2 > len(decs) else 'red'),
                     full=dict(n=full['evaluated']['n'], delta=full['evaluated']['delta'], z=full['evaluated']['z'], cap=full['evaluated']['c_cap_rate'], decision=full['decision'], h=full['evaluated']['h_pass'], c=full['evaluated']['c_pass']),
                     batch=full['batch_pre_post'], batch_overlap=statistics.mean(d['batch_overlap'] for d in decs),
                     expected=('red' if e in EXPECT_RED else 'green'), reasons=decs[0]['reasons']))
json.dump(rows, open(os.path.join(os.path.dirname(__file__), 'backtest_rows.json'), 'w'), indent=1)
print("| ep | H→C | H n / C n | eval n (S\\batch) | z mean [min,max] | C cap | green/seeds | gate | expected | full-pool Δ, z, cap | batch pre→post (analysis) |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    if 'skip' in r: print("| %d | — | | | | | | skip: %s | | | |" % (r['epoch'], r['skip'])); continue
    f = r['full']; b = r['batch']
    ok = '' if r['decision'] == r['expected'] else ' ✗'
    print("| %d | %s→%s | %d%s / %d%s | %.0f | %+.2f [%+.2f,%+.2f] | %.3f | %d/%d | **%s**%s | %s | %+.3f, %+.1f, %.3f | %s→%s (n=%d) |" % (
        r['epoch'], r['H'], r['C'], r['h_n'], 'p' if r['h_partial'] else '', r['c_n'], 'p' if r['c_partial'] else '', r['n_eval'], r['z_mean'], r['z_min'], r['z_max'], r['cap'],
        round(r['green_frac'] * len(SEEDS)), len(SEEDS), r['decision'].upper(), ok, r['expected'], f['delta'] or 0, f['z'], f['cap'] or 0,
        None if b['h_pass'] is None else round(b['h_pass'], 3), None if b['c_pass'] is None else round(b['c_pass'], 3), b['n']))
ev = [r for r in rows if 'skip' not in r]
healthy = [r for r in ev if r['expected'] == 'green']; bad = [r for r in ev if r['expected'] == 'red']
fr = sum(1 - r['green_frac'] for r in healthy) / max(1, len(healthy))
tr = sum(1 - r['green_frac'] for r in bad) / max(1, len(bad))
print("\nhealthy steps: %d, per-seed false-reject rate %.3f; majority false rejects: %s" % (len(healthy), fr, [r['epoch'] for r in healthy if r['decision'] == 'red']))
print("expected-bad steps: %d, per-seed reject rate %.3f; majority misses: %s" % (len(bad), tr, [r['epoch'] for r in bad if r['decision'] == 'green']))
