import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("k1s", sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
recs = m.verdicts(None)
turns = [r for r in recs if r.get('status') == 'turn']
both = [r for r in turns if r.get('h_pass') is not None and r.get('c_pass') is not None]
print("turns", len(turns), "with both pass rates", len(both))
for r in turns:
    if r.get('h_pass') is None: print("missing", r['epoch_index'], r.get('verdict'), r.get('pass_basis'), r.get('pass_note'))
# cross-check against public reveals
W = m.WEB
bad = 0; chk = 0
for r in turns:
    if r.get('pass_basis') != 'heldout' or r.get('c_pass') is None: continue
    rv = m.rj(W / 'gate' / 'reveal' / ('%d.json' % r['epoch_index']))
    rr = m.rj(W / 'gate' / 'heldout' / (r['heldout_round'] + '.json'))
    if not rv or not rr: print("no public reveal for", r['epoch_index']); continue
    kept = set(str(k) for k in (rr['baseline'].get('kept') or []))
    bt = set(str(k) for k in (rv.get('batch_tasks') or []))
    cp = rv['candidate_per_task']; ks = [k for k in kept if k in cp and k not in bt]
    c = sum(cp[k]['c'] for k in ks); n = sum(cp[k]['n'] for k in ks)
    chk += 1
    if abs(c / n - r['c_pass']) > 1e-9 or len(ks) != r['n']: bad += 1; print("mismatch", r['epoch_index'], c / n, r['c_pass'], len(ks), r['n'])
print("public cross-check", chk, "mismatches", bad)
json.dump(recs, open('/tmp/pcf-recs.json', 'w'))
