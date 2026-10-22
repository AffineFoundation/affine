"""Offline tests for weights_v9 (throwaway keys; no chain calls)."""
import base64, importlib.util, json, math, sys, tempfile, unittest
from pathlib import Path
from nacl.signing import SigningKey
spec = importlib.util.spec_from_file_location('w9', str(Path(__file__).with_name('weights_v9.py'))); w = importlib.util.module_from_spec(spec); spec.loader.exec_module(w)
AK, SK = SigningKey.generate(), SigningKey.generate(); AUTH = AK.verify_key.encode().hex(); SCOR = SK.verify_key.encode().hex()
def sign(p, k): return dict(payload=p, signer=k.verify_key.encode().hex(), signature=base64.b64encode(k.sign(w.canonical(p)).signature).decode())
OWNER = '5OWNER'
class T(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp()); (self.d / 'scores').mkdir()
        self.cfg = dict(version=w.CONFIG_VERSION, sequence=1, issued_at=0, mode='v9-only-burn-uid0', miner_share=1.0, max_uid_share=None,
                        scorer_pubkey=SCOR, scores_dir=str(self.d / 'scores'), log_dir=str(self.d / 'log'), max_score_age_epochs=2, max_flag_rate=0.05, note='t')
    def write(self, cfg=None, gate=None, E=70, placebo=0, flag=0.01):
        (self.d / 'c.json').write_text(json.dumps(sign(cfg or self.cfg, AK)))
        gate = gate if gate is not None else {'pkA': dict(gated=True, V=3.0), 'pkB': dict(gated=True, V=1.0), 'pkC': dict(gated=False, V=0.0)}
        p = dict(version=w.SCORE_VERSION, epoch=E, seed=dict(seed_block=1), window_gate=gate,
                 window_summary=dict(scored_uids=len(gate), flag_rate_stouffer_uncorrected=flag), false_positive_checks=dict(label_shuffle=dict(gated_max=placebo)))
        (self.d / 'scores' / f'score-e{E}.json').write_text(json.dumps(sign(p, SK)))
        return str(self.d / 'c.json')
    def vec(self, L=70, **k): return w.v9_vector(w.load_config(self.write(**k), AUTH), L)
    def test_proportional(self):
        sh, burn, info = self.vec(); self.assertAlmostEqual(sh['pkA'], 0.75); self.assertAlmostEqual(sh['pkB'], 0.25); self.assertAlmostEqual(burn, 0.0)
    def test_cap_burns_excess(self):
        c = dict(self.cfg, max_uid_share=0.5); sh, burn, _ = self.vec(cfg=c); self.assertAlmostEqual(sh['pkA'], 0.5); self.assertAlmostEqual(burn, 0.25)
    def test_none_gated_full_burn(self):
        sh, burn, info = self.vec(gate={'pkC': dict(gated=False, V=0.0)}); self.assertEqual(sh, {}); self.assertEqual(burn, 1.0)
    def test_stale_full_burn(self):
        sh, burn, info = self.vec(L=73); self.assertEqual(burn, 1.0); self.assertIn('score_missing_or_stale', info['reasons'])
    def test_placebo_and_flag_alarm(self):
        self.assertEqual(self.vec(placebo=1)[1], 1.0); self.assertEqual(self.vec(flag=0.2)[1], 1.0)
    def test_bad_scorer_signature(self):
        p = self.write(); f = self.d / 'scores' / 'score-e70.json'; d = json.loads(f.read_text()); d['payload']['epoch'] = 70; d['payload']['window_gate']['pkB']['V'] = 99.0
        f.write_text(json.dumps(d)); sh, burn, info = w.v9_vector(w.load_config(p, AUTH), 70); self.assertEqual(burn, 1.0)
    def test_recipients_and_burn_entry(self):
        p = self.write(); r = w.make(p, AUTH, OWNER, None) if False else None
        c = w.load_config(p, AUTH); sh, burn, info = w.v9_vector(c, 70)
        # use the recipients closure directly (adapter patch is exercised on the server against the real ChainAdapter)
        import types
        rep = w.make.__code__  # noqa
        regs = {'hkA': dict(public_key='pkA', uid=5), 'hkC': dict(public_key='pkC', uid=7)}
        fns = make_fns(p)
        a = fns['calculate'](dict(shares=sh, burn=burn, info=info, config_sequence=1), {}, 3600); fns['validate_assessment'](a, None, 3600)
        pts, sel, exc = fns['recipients'](a, regs)
        self.assertEqual(exc, ['pkB']); self.assertEqual(set(pts), {'hkA', OWNER}); self.assertAlmostEqual(pts[OWNER] / sum(pts.values()), 0.25, places=6)
    def test_full_burn_recipients(self):
        p = self.write(gate={}); fns = make_fns(p); c = w.load_config(p, AUTH); sh, burn, info = w.v9_vector(c, 70)
        a = fns['calculate'](dict(shares=sh, burn=burn, info=info, config_sequence=1), {}, 3600); pts, sel, exc = fns['recipients'](a, {})
        self.assertEqual(pts, {OWNER: w.UNITS})
    # ---- seq5 always-pay ----
    PO = dict(rule=w.PAYOUT_RULE, temperature=1.0, top_k=2, min_units=3, fallback_top_n=1)
    def g(self, **zs): return {m: dict(gated=False, V=0.0, B=b, stouffer_z=z) for m, (z, b) in zs.items()}
    def test_alwayspay_softmax_topk(self):
        sh, burn, info = self.vec(cfg=dict(self.cfg, payout=self.PO), gate=self.g(pkA=(2.0, 5), pkB=(1.0, 5), pkC=(0.5, 5), pkD=(9.0, 2)))
        self.assertEqual(set(sh), {'pkA', 'pkB'}); self.assertAlmostEqual(sh['pkA'], math.exp(1) / (1 + math.exp(1))); self.assertAlmostEqual(burn, 0.0)
        self.assertEqual(info['eligible'], 3); self.assertEqual(info['reasons'], [])
    def test_alwayspay_no_positive_fallback(self):
        sh, burn, info = self.vec(cfg=dict(self.cfg, payout=self.PO), gate=self.g(pkA=(-0.5, 4), pkB=(-0.1, 4)))
        self.assertEqual(sh, {'pkB': 1.0}); self.assertTrue(info['fallback_no_positive'])
    def test_alwayspay_alarms_diagnostic_only(self):
        sh, burn, info = self.vec(cfg=dict(self.cfg, payout=self.PO), gate=self.g(pkA=(1.0, 4)), placebo=3, flag=0.5)
        self.assertEqual(sh, {'pkA': 1.0}); self.assertEqual(info['diagnostic_alarms'], ['placebo_gated', 'calibration_flag_rate'])
    def test_alwayspay_stale_still_burns(self):
        sh, burn, info = self.vec(L=73, cfg=dict(self.cfg, payout=self.PO), gate=self.g(pkA=(1.0, 4))); self.assertEqual(burn, 1.0)
    def test_alwayspay_none_eligible_burns(self):
        sh, burn, info = self.vec(cfg=dict(self.cfg, payout=self.PO), gate=self.g(pkA=(3.0, 2))); self.assertEqual(burn, 1.0); self.assertIn('no_eligible_miner', info['reasons'])
    def test_alwayspay_bad_payout_rejected(self):
        with self.assertRaises(ValueError): self.vec(cfg=dict(self.cfg, payout=dict(self.PO, top_k=0)))
    def test_alwayspay_unregistered_renormalised(self):
        c = dict(self.cfg, payout=self.PO); p = self.write(cfg=c, gate=self.g(pkA=(2.0, 5), pkB=(1.0, 5)))
        fns = make_fns(p); sh, burn, info = w.v9_vector(w.load_config(p, AUTH), 70)
        a = fns['calculate'](dict(shares=sh, burn=burn, info=info, config_sequence=1), {}, 3600); fns['validate_assessment'](a, None, 3600)
        pts, sel, exc = fns['recipients'](a, {'hkA': dict(public_key='pkA', uid=5)})
        self.assertEqual(exc, ['pkB']); self.assertEqual(set(pts), {'hkA'}); self.assertEqual(pts['hkA'], w.UNITS)
class TS(T):
    # ---- seq6 share-EMA smoothing ----
    SM = dict(rule=w.SMOOTH_RULE, half_life_epochs=4, lookback_epochs=12, min_share=0.01)
    def cfgs(self, **k): return dict(self.cfg, payout=dict(T.PO, top_k=10), smoothing=dict(self.SM, **k))
    def hist(self, gates, c):   # gates: {E: gate}; writes config + every score
        for E, g in gates.items(): self.write(cfg=c, gate=g, E=E)
        return w.load_config(str(self.d / 'c.json'), AUTH)
    def test_no_smoothing_is_seq5_exact(self):
        c0 = dict(self.cfg, payout=dict(T.PO, top_k=10)); g = self.g(pkA=(2.0, 5), pkB=(1.0, 5))
        cA = self.hist({69: self.g(pkC=(5.0, 5)), 70: g}, c0); a = w.v9_vector(cA, 70)
        self.assertNotIn('smoothing', a[2]); self.assertAlmostEqual(a[0]['pkA'], math.exp(1) / (1 + math.exp(1))); self.assertNotIn('pkC', a[0])
    def test_steady_state_equals_unsmoothed(self):
        g = self.g(pkA=(2.0, 5), pkB=(1.0, 5)); c = self.hist({E: g for E in range(60, 71)}, self.cfgs())
        sh, burn, info = w.v9_vector(c, 70); self.assertAlmostEqual(sh['pkA'], math.exp(1) / (1 + math.exp(1))); self.assertAlmostEqual(burn, 0.0)
        self.assertEqual(info['smoothing']['epochs_used'], list(range(70, 58, -1))[:11]); self.assertEqual(info['smoothing']['epochs_skipped'], [59])
    def test_ramp_and_decay_halflife(self):
        # pkA alone for 11 epochs, then pkN alone appears (eligible); pkA stays eligible but unpaid in the raw rule
        a = 1 - 2 ** -0.25; gs = {E: self.g(pkA=(3.0, 5), pkN=(-1.0, 5)) for E in range(59, 70)}; gs[70] = self.g(pkA=(-1.0, 5), pkN=(3.0, 5))
        c = self.hist(gs, self.cfgs()); sh, _, info = w.v9_vector(c, 70)
        W = sum(a * (1 - a) ** k for k in range(12)); self.assertAlmostEqual(sh['pkN'], a / W, places=9); self.assertAlmostEqual(sh['pkA'] + sh['pkN'], 1.0)
    def test_stopped_miner_cut_when_ineligible(self):
        gs = {E: self.g(pkA=(3.0, 5), pkB=(1.0, 5)) for E in range(65, 70)}; gs[70] = self.g(pkB=(1.0, 5), pkA=(3.0, 2))   # pkA B < min_units now
        sh, _, info = w.v9_vector(self.hist(gs, self.cfgs()), 70); self.assertEqual(set(sh), {'pkB'}); self.assertAlmostEqual(sh['pkB'], 1.0)
        self.assertEqual(info['smoothing']['dropped_ineligible'], 1)
    def test_unscored_miner_cut(self):
        gs = {E: self.g(pkA=(3.0, 5), pkB=(1.0, 5)) for E in range(65, 70)}; gs[70] = self.g(pkB=(1.0, 5))
        sh, _, _ = w.v9_vector(self.hist(gs, self.cfgs()), 70); self.assertEqual(set(sh), {'pkB'})
    def test_lookback_bound(self):
        gs = {60: self.g(pkA=(9.0, 5), pkB=(1.0, 5))}; gs.update({E: self.g(pkA=(-1.0, 5), pkB=(1.0, 5)) for E in range(61, 71)})
        sh, _, _ = w.v9_vector(self.hist(gs, self.cfgs(lookback_epochs=10, min_share=0.0)), 70); self.assertEqual(set(sh), {'pkB'})
    def test_min_share_prunes_and_renormalises(self):
        gs = {E: self.g(pkA=(3.0, 5), pkB=(1.0, 5)) for E in range(59, 70)}; gs[70] = self.g(pkA=(-2.0, 5), pkB=(3.0, 5), pkC=(-3.0, 5))
        c = self.hist(gs, self.cfgs(min_share=0.5 - 1e-9)); sh, _, info = w.v9_vector(c, 70)
        self.assertEqual(len(sh), 1); self.assertAlmostEqual(sum(sh.values()), 1.0)
    def test_bad_older_score_skipped(self):
        gs = {E: self.g(pkA=(3.0, 5), pkB=(1.0, 5)) for E in range(66, 71)}; c = self.hist(gs, self.cfgs())
        f = self.d / 'scores' / 'score-e68.json'; d = json.loads(f.read_text()); d['payload']['window_gate']['pkB']['stouffer_z'] = 99.0; f.write_text(json.dumps(d))
        sh, _, info = w.v9_vector(c, 70); self.assertIn(68, info['smoothing']['epochs_skipped']); self.assertGreater(sh['pkA'], sh['pkB'])
    def test_stale_newest_still_burns(self):
        gs = {E: self.g(pkA=(3.0, 5)) for E in range(66, 71)}; sh, burn, info = w.v9_vector(self.hist(gs, self.cfgs()), 73); self.assertEqual(burn, 1.0)
    def test_bad_smoothing_rejected(self):
        for bad in (dict(half_life_epochs=0), dict(lookback_epochs=0), dict(min_share=0.6), dict(rule='x')):
            with self.assertRaises(ValueError): self.hist({70: self.g(pkA=(1.0, 5))}, self.cfgs(**bad))
        with self.assertRaises(ValueError): self.hist({70: self.g(pkA=(1.0, 5))}, dict(self.cfg, smoothing=self.SM))   # smoothing needs payout
    def test_assessment_json_canonical(self):
        gs = {E: self.g(pkA=(3.0, 5), pkB=(1.0, 5)) for E in range(66, 71)}; p = self.d / 'c.json'; c = self.hist(gs, self.cfgs())
        fns = make_fns(str(p)); sh, burn, info = w.v9_vector(c, 70)
        a = fns['calculate'](dict(shares=sh, burn=burn, info=info, config_sequence=3), {}, 3600); fns['validate_assessment'](a, None, 3600); w.canonical(a)
def make_fns(p): return w.make_rule(p, AUTH, OWNER)
if __name__ == '__main__': unittest.main(verbosity=1)
