#!/usr/bin/env python3
"""First-view slices of the big site JSON files (Jacob pt1076u, 2026-11-02). Read-only projection: every value is copied
unchanged from the source file; nothing is recomputed. Written atomically next to the source, only when the source changed.
  site-metrics-recent.json  every epoch's index + groups_total (header Groups total), and the timestamp fields of the
                            last 60 epochs (Epoch duration tile shows the last 20 finalized)      ~726 KB -> ~25 KB raw
  gate-recent.json          gate entries for those same epochs, only epoch/verdict/head/timestamps  ~643 KB -> ~20 KB raw
  reg-cost-recent.json      reg-cost.json with the last 35 days of hourly points (+ the point in force before) for the
                            default 7D view; the page loads the full file in the background for 30D/1Y/All
"""
import json, os, sys, time
W = sys.argv[1] if len(sys.argv) > 1 else '/home/const/subnet120/affine/website'
KEEP = 60
TF = ('index', 'phase', 'start', 'deadline', 'sealed_at', 'training_started_at', 'training_completed_at', 'finalized_at', 'groups_total')
def src(n): return os.path.join(W, n)
def stale(out, *ins):
    try: o = os.stat(src(out)).st_mtime
    except FileNotFoundError: return True
    return any(os.stat(src(i)).st_mtime > o for i in ins)
def write(name, obj):
    tmp = src('.' + name + '.tmp')
    with open(tmp, 'w') as f: json.dump(obj, f, separators=(',', ':'))
    os.chmod(tmp, 0o644); os.replace(tmp, src(name))
def load(n):
    with open(src(n)) as f: return json.load(f)
def main():
    if stale('site-metrics-recent.json', 'site-metrics.json') or stale('gate-recent.json', 'site-metrics.json', 'gate.json'):
        sm = load('site-metrics.json'); ep = sorted((e for e in sm.get('epochs') or [] if isinstance(e.get('index'), int)), key=lambda e: e['index'])
        recent = {e['index'] for e in ep[-KEEP:]}
        out = [{k: e[k] for k in (TF if e['index'] in recent else ('index', 'groups_total')) if k in e} for e in ep]
        write('site-metrics-recent.json', {'version': sm.get('version'), 'generated_at': sm.get('generated_at'), 'slice': 'epochs: index+groups_total for all, timestamps for the last %d; full file: /site-metrics.json' % KEEP, 'epochs': out})
        g = load('gate.json'); lo = min(recent) if recent else 0
        ent = [{k: e[k] for k in ('epoch', 'verdict', 'head', 'timestamps') if k in e} for e in g.get('entries') or [] if isinstance(e.get('epoch'), int) and e['epoch'] >= lo]
        write('gate-recent.json', {'version': g.get('version'), 'updated_at': g.get('updated_at'), 'slice': 'entries from E%d, fields epoch/verdict/head/timestamps; full file: /gate.json' % lo, 'entries': ent})
    if stale('reg-cost-recent.json', 'reg-cost.json'):
        r = load('reg-cost.json'); cut = (r.get('generated_at') or time.time()) - 35 * 86400
        for k in ('sn120', 'network'):
            o = r.get(k)
            if isinstance(o, dict) and isinstance(o.get('points'), list):
                P = o['points']; i = next((j for j, p in enumerate(P) if isinstance(p, list) and p and p[0] >= cut), len(P))
                o['points'] = P[max(0, i - 1):]
        r['slice'] = 'last 35 days of points (+ the one in force before); full file: /reg-cost.json'
        write('reg-cost-recent.json', r)
if __name__ == '__main__':
    main()
