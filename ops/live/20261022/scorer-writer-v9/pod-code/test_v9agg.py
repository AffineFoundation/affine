"""Unit tests for v9agg (synthetic data only)."""
import random, numpy as np, math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v9agg, v9common
K = 12
CH = [dict(kind='subject', subject='s%d' % (k % 3), half=k % 2, c=k, tasks=[100 + 4 * k + j for j in range(4)]) for k in range(K)]

def mk(rng, miner, slot, task, shift=0.0):
    r = list(rng.choice([0.0, 1.0], 8)); r[0], r[1] = 1.0, 0.0
    d = rng.normal(0, 1, (8, K)) * 8192; d[np.array(r) > 0] += shift * 8192
    return dict(miner=miner, uid=hash(miner) % 256, slot=slot, task=task, rewards=r, dots=d.tolist())

def test_leave_task_out():
    rng = np.random.default_rng(0); g = mk(rng, 'a', 0, CH[0]['tasks'][0])
    D = np.array(g['dots']); D[np.array(g['rewards']) > 0, 0] += 1e9                 # inflate only the chunk that contains the group's own task
    s1 = v9agg.group_score(g['rewards'], D, g['task'], CH); s0 = v9agg.group_score(g['rewards'], g['dots'], g['task'], CH)
    assert abs(s1 - s0) < 1e-9, 'own-task chunk must be dropped'
    s2 = v9agg.group_score(g['rewards'], D, 9999, CH); assert s2 > s0 + 1, 'other tasks keep chunk 0'

def test_zero_variance_zero_score():
    assert v9agg.group_score([1.0] * 8, np.ones((8, K)), 5, CH) == 0.0

def test_null_p_calibrated():
    rng = np.random.default_rng(1); cal = rng.normal(size=4000); nl = v9agg.Null(cal, rng)
    M = rng.normal(size=(20000, 3)).max(1); p = np.array([nl.p(m) for m in M])
    assert abs((p < 0.05).mean() - 0.05) < 0.01 and abs((p < nl.q).mean() - nl.q) < 0.01, (p < 0.05).mean()

def _epoch(rng, nminers=200, planted=(), shift=0.6, slots=9):
    G = []
    for i in range(nminers):
        m = 'm%03d' % i
        for s in range(slots): G.append(mk(rng, m, s, int(rng.integers(0, 7000)), shift if (m in planted and s % 3 == 0) else 0.0))
    return G

def test_honest_fp_and_power():
    rng = np.random.default_rng(2); fold = lambda m: v9common.HS('x', 'fold', m) % 2
    planted = {'m000', 'm001', 'm002'}; per = []
    for e in range(4):
        miners, info, n = v9agg.score_epoch(_epoch(rng, planted=planted), CH, fold, rng); per.append(miners)
    out, summ = v9agg.gate_window(per)
    honest_gated = sum(r['gated'] for m, r in out.items() if m not in planted)
    assert honest_gated == 0, honest_gated
    assert all(out[m]['gated'] for m in planted), [out[m] for m in planted]
    assert summ['flag_rate_stouffer_uncorrected'] < 0.05

def test_splitting_gains_nothing():
    """Same groups re-slotted into fewer/more units: an honest miner cannot raise its gate statistic by splitting,
    because every unit is a fixed 3-slot unit against the same 3-group null (partial units are only conservative)."""
    rng = np.random.default_rng(3); fold = lambda m: 1 if m == 'x' else int(m[3:]) % 2
    G = [mk(rng, 'cal%03d' % i, s, int(rng.integers(0, 7000))) for i in range(150) for s in range(9)]
    hon = [mk(rng, 'x', s, int(rng.integers(0, 7000))) for s in range(9)]
    miners, info, _ = v9agg.score_epoch(G + hon, CH, fold, rng)
    assert len(miners['x']['units']) == 3 and all(u['groups'] == 3 for u in miners['x']['units'])
    one = [dict(g, slot=0) if i == 0 else None for i, g in enumerate(hon)]
    m2, _, _ = v9agg.score_epoch(G + [one[0]], CH, fold, rng)
    assert m2['x']['units'][0]['groups'] == 1   # a 1-group submission is one partial unit, same null
    nl = v9agg.Null([g for g in [v9agg.group_score(x['rewards'], x['dots'], x['task'], CH) for x in G if int(x['miner'][3:]) % 2 == 0]], rng)
    s = v9agg.group_score(hon[0]['rewards'], hon[0]['dots'], hon[0]['task'], CH)
    assert m2['x']['units'][0]['p'] >= 1 - (np.searchsorted(nl.cal, s) / len(nl.cal)) ** 3 - 1e-12

def test_seed_rule():
    s = dict(rule='sn120-v9-seed-rule-v1', epoch=70, epoch_id='e70', deadline=100, anchor_block=10, anchor_ts_ms=100000, margin_blocks=10, seed_block=20,
             seed_block_hash='0xab')
    import hashlib; s['seed'] = hashlib.sha256(b'sn120-v9-ref-seed-v1:e70:0xab').hexdigest()
    v9common.check_seed(dict(s), 70)
    for bad in (dict(s, seed_block=15), dict(s, anchor_ts_ms=99999), dict(s, seed='0' * 64), dict(s, epoch=69)):
        try: v9common.check_seed(bad, 70); raise AssertionError('accepted bad seed')
        except ValueError: pass

def test_draw_depends_only_on_seed():
    T = [dict(data=dict(subject='s%d' % (i % 7))) for i in range(7000)]; pool = list(range(0, 7000, 1))
    a = v9common.draw_reference('aa', pool, T); b = v9common.draw_reference('aa', list(reversed(pool)), T); c = v9common.draw_reference('ab', pool, T)
    assert a == b and a != c and len(a) == 1400

def test_p_open_interval_finite_stouffer():
    from scipy.stats import norm
    rng = np.random.default_rng(7); nl = v9agg.Null(rng.normal(size=500), rng)
    lo, hi = nl.p(1e9), nl.p(-1e9)
    assert 0 < lo < hi < 1 and np.isfinite(norm.isf(hi)) and np.isfinite(norm.isf(lo))
    g, _ = v9agg.gate_window([{'m': dict(uid=1, units=[dict(p=hi, q=nl.q, hit=False, sum_z=-9.0, groups=3)])}])
    assert np.isfinite(g['m']['stouffer_z'])

if __name__ == '__main__':
    n = 0
    for k, f in sorted(globals().items()):
        if k.startswith('test_'): f(); n += 1; print('ok', k, flush=True)
    print('ALL PASS', n)
