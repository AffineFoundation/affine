#!/usr/bin/env python3
"""SN120 king gain multiplier (RA/20261102-gain-mult; Jacob pt1136u/pt1137u 2026-11-02: live, lambda = 1, only kings crowned after the
activation epoch, no backfill).

For each GREEN king crowned at epoch > activation_epoch (candidate = king, DECISION head = parent):
  per task (both heads sampled on it, same token-cap regime): d = king pass rate - parent pass rate (pooled over every prior-sampler pass)
  gain = mean(d), SE = sd(d)/sqrt(n), cons = max(0, gain - SE)
  settled iff n >= min_shared; else m = 1.
  median = median of the last `ref_window` POSITIVE cons values over (the frozen pre-activation analysis kings, then settled post-activation
  kings), in crowning order. If the median is undefined or <= 0: m = 1 for everyone.
  m = clamp(1 + lambda * (cons / median - 1), m_min, m_max)
Kings crowned at or before activation_epoch: m = 1 permanently (they are not in the output; k1_service defaults to 1).
Fail-safe: if the gate's prior cap no longer matches the last configured cap regime, no recomputation: the last good private state is kept
(unsettled kings stay m = 1) and the reason is recorded.
Writes STATE/GAIN-MULT.private.json (0600: gains, SE, n) and STATE/GAIN-MULT.public.json (0644: per-king m rounded + settled only).
usage: gain_mult.py [--config CONFIG] [--dry]   (--dry: print, write nothing)"""
import json, math, os, statistics, sys, time
from pathlib import Path

RA = Path('/home/const/subnet120-rewrite/state/root-audits')
V = RA / '20261102-gain-mult'
CFG_PATH = Path(sys.argv[sys.argv.index('--config') + 1]) if '--config' in sys.argv else V / 'code' / 'config.json'
DRY = '--dry' in sys.argv
ST = V / 'state'
PRIV, PUB = ST / 'GAIN-MULT.private.json', ST / 'GAIN-MULT.public.json'
GRADES = RA / '20261101-prior-table/private/table/grades.private.jsonl'
GATE = RA / '20261026-promotion-gate/state/epochs'
PG_FLAG = RA / '20261023-toploc-credit-train-gate/staged/PROMOTION-GATE.json'


def rj(p, default=None):
    try: return json.loads(Path(p).read_text())
    except Exception: return default


def wj(p, obj, mode):
    p = Path(p); tmp = p.with_name(p.name + '.tmp')
    with open(tmp, 'w') as f: json.dump(obj, f, indent=1, sort_keys=True)
    os.chmod(tmp, mode); os.replace(tmp, p)


def cr(ts=None):
    return time.strftime('%Y-%m-%d %H:%M CR (UTC-6)', time.gmtime((ts or time.time()) - 6 * 3600))


def regime_of(t, regimes):
    r = None
    for g in regimes:
        if t >= float(g['from_t']): r = g['cap']
    return r


def kings(cfg):
    """GREEN epochs > activation_epoch: [(epoch, king, parent)] (same verdict source as k1_service: DECISION, REGATE supersedes)."""
    out = []
    a = int(cfg['activation_epoch'])
    for p in sorted((p for p in GATE.iterdir() if p.name.isdigit() and int(p.name) > a), key=lambda p: int(p.name)):
        d = rj(p / 'DECISION.json') or {}
        v = d.get('decision')
        rg = rj(p / 'REGATE.json') or {}
        if rg.get('decision') in ('green', 'red') and rg.get('candidate') == d.get('candidate'): v = rg['decision']
        if v == 'green' and d.get('candidate') and d.get('head'):
            out.append((int(p.name), d['candidate'], d['head']))
    return out


def aggregate(heads, regimes):
    """-> {(head, cap): {task_id: [c, n]}} for the given heads, from the prior-sampler grades (raw counts, no discount)."""
    agg = {}
    keys = {h: h for h in heads}
    with open(GRADES) as f:
        for line in f:
            h = None
            for k in keys:
                if k in line: h = k; break
            if h is None: continue
            try: r = json.loads(line)
            except Exception: continue
            if r.get('head') != h or not r.get('n'): continue
            cap = regime_of(float(r['t']), regimes)
            a = agg.setdefault((h, cap), {}).setdefault(r['task_id'], [0, 0])
            a[0] += int(r.get('c') or 0); a[1] += int(r['n'])
    return agg


def pair(agg, king, parent, regimes, se_mult=1.0):
    """Best same-cap pairing (the cap regime with the most shared tasks) -> dict(n, gain, se, cons, cap)."""
    best = None
    for cap in sorted({g['cap'] for g in regimes}):
        K, P = agg.get((king, cap)) or {}, agg.get((parent, cap)) or {}
        sh = [t for t in K if t in P and K[t][1] > 0 and P[t][1] > 0]
        if best is None or len(sh) > best[1]: best = (cap, len(sh), sh, K, P)
    cap, n, sh, K, P = best
    if n < 2: return dict(n=n, cap=cap, gain=None, se=None, cons=None)
    d = [K[t][0] / K[t][1] - P[t][0] / P[t][1] for t in sh]
    g = math.fsum(d) / n
    se = math.sqrt(math.fsum((x - g) ** 2 for x in d) / (n - 1)) / math.sqrt(n)
    return dict(n=n, cap=cap, gain=g, se=se, cons=max(0.0, g - se_mult * se))


def compute(cfg):
    regimes = cfg['cap_regimes']
    ks = kings(cfg)
    heads = sorted({h for _, k, p in ks for h in (k, p)})
    agg = aggregate(heads, regimes) if heads else {}
    rows = []
    for e, k, p in ks:
        r = pair(agg, k, p, regimes, float(cfg.get('se_mult', 1.0)))
        r.update(epoch=e, king=k, parent=p, settled=r['n'] >= int(cfg['min_shared']) and r['cons'] is not None)
        rows.append(r)
    ref = [float(x['cons']) for x in cfg['reference_frozen']] + [r['cons'] for r in rows if r['settled']]
    pos = [x for x in ref if x > 0][-int(cfg['ref_window']):]
    med = statistics.median(pos) if pos else None
    lam, lo, hi = float(cfg['lambda']), float(cfg['m_min']), float(cfg['m_max'])
    for r in rows:
        if not r['settled']: r.update(m=1.0, why='n_shared %d < %d (m = 1 until settled)' % (r['n'], int(cfg['min_shared'])))
        elif not med or med <= 0: r.update(m=1.0, why='median reference undefined or 0 (m = 1)')
        else: r.update(m=min(hi, max(lo, 1.0 + lam * (r['cons'] / med - 1.0))), why=None)
    return rows, dict(median=med, positive_reference=pos, n_reference=len(pos))


def main():
    cfg = rj(CFG_PATH)
    if not cfg or cfg.get('enabled') is not True:
        print(json.dumps(dict(event='disabled'))); return
    now = time.time()
    flag = rj(PG_FLAG) or {}
    gate_cap = ((flag.get('heldout') or {}).get('prior_v1') or {}).get('max_tokens')
    last_cap = cfg['cap_regimes'][-1]['cap']
    prev = rj(PRIV) or {}
    if gate_cap is not None and int(gate_cap) != int(last_cap):
        why = 'gate prior cap %s != last configured cap regime %s: kept the last state (add a cap_regimes entry to resume)' % (gate_cap, last_cap)
        doc = dict(prev or dict(kings=[]), frozen=True, frozen_reason=why, checked_at=now, checked_at_cr=cr(now))
        rows, ref = doc.get('kings') or [], doc.get('reference') or {}
    else:
        rows, ref = compute(cfg)
        doc = dict(frozen=False, kings=rows, reference=ref, checked_at=now, checked_at_cr=cr(now))
    doc.update(version='sn120-king-gain-mult-v1', activation_epoch=cfg['activation_epoch'], params={k: cfg[k] for k in
               ('lambda', 'min_shared', 'm_min', 'm_max', 'ref_window', 'se_mult')}, generated_at=now, generated_at_cr=cr(now))
    pub = dict(version='sn120-king-gain-mult-public-v1', generated_at=now, generated_at_cr=cr(now), enabled=True,
               activation_epoch=cfg['activation_epoch'], lambda_=cfg['lambda'], frozen=doc.get('frozen', False),
               rule=cfg['public_rule'],
               kings=[dict(epoch=r['epoch'], king=r['king'][:12], m=round(float(r['m']), 3), settled=bool(r['settled'])) for r in rows])
    pub['lambda'] = pub.pop('lambda_')
    if DRY:
        print(json.dumps(dict(private=doc, public=pub), default=str, indent=1)); return
    ST.mkdir(parents=True, exist_ok=True); os.chmod(ST, 0o700)
    wj(PRIV, doc, 0o600)
    wj(PUB, pub, 0o644)
    print(json.dumps(dict(event='gain_mult', kings=[(r['epoch'], round(r['m'], 3), r['settled'], r['n']) for r in rows],
                          median=ref.get('median'), frozen=doc.get('frozen'), at_cr=cr(now))))


if __name__ == '__main__':
    main()
