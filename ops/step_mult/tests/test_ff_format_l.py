"""Unit tests: ff_format + affine_batch train.step_mult + v8.3.3l knobs (RA/20261102-step-mult); v1/v2 groups unchanged vs the live base.
usage: test_ff_format_sm.py BASE_DIR NEW_DIR"""
import importlib.util, json, sys, unittest
BASE, NEW = sys.argv[1], sys.argv[2]; del sys.argv[1:3]
def load(d, n, name):
    spec = importlib.util.spec_from_file_location(name, d + '/' + n); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
F0 = load(BASE, 'ff_format.py', 'ff_base'); F = load(NEW, 'ff_format.py', 'ff_new'); AB = load(NEW, 'affine_batch.py', 'ab_new')
TOK = F.tokenizer('/home/const/subnet120-rewrite/state/root-audits/20261030-determinism/tok')
HK = '5C4khFWyExampleHotkeyxxxxxxxxxxxxxxxxxxxxxxxxxx'
G = [dict(prompt='What is 2+2?', rollouts=[dict(output='4', reward=1.0), dict(output='5', reward=0)])]
CAPS = dict(max_tasks_per_miner=125, max_rollouts_per_miner=1000)
def raw(train=None, version=None):
    d = dict(version=version or F.FORMAT_V2, hotkey=HK, groups=G)
    if train is not None: d['train'] = train
    return json.dumps(d).encode()
class T(unittest.TestCase):
    def ok(self, b):
        ok, info, groups = F.check(b, HK, TOK, 125, 1000, True); self.assertTrue(ok, info); return info, groups
    def bad(self, b, frag):
        ok, info, _ = F.check(b, HK, TOK, 125, 1000, True); self.assertFalse(ok); self.assertIn(frag, info)
        ok2, msg = AB.check(b, HK, CAPS); self.assertFalse(ok2, msg)
    def test_old_files_same_groups_and_defaults(self):
        for b in (raw(version=F.FORMAT), raw(), raw({}), raw({'lr': 8e-07}), raw({'kl': 0.01, 'lr': 2.5e-07})):
            a = F0.validate(b, HK, TOK, 125, 1000, True); n = F.validate(b, HK, TOK, 125, 1000, True)
            self.assertEqual(a['groups'], n['groups'])
            exp = dict(a['summary']['train'], step_mult=1)
            if b'"train"' in b: exp.update(F.KNOB_DEFAULTS)     # v2: effective knobs = defaults
            self.assertEqual(n['summary']['train'], exp)
            self.assertEqual(a['summary']['train_given'], n['summary']['train_given'])
            self.assertTrue(AB.check(b, HK, CAPS)[0])
    def test_step_mult_ok(self):
        for k in range(1, 9):
            info, _ = self.ok(raw({'step_mult': k}))
            self.assertEqual(info['train'], dict(lr=5e-07, kl=0.02, step_mult=k, **F.KNOB_DEFAULTS)); self.assertEqual(info['train_given'], ['step_mult'])
            ok, msg = AB.check(raw({'step_mult': k}), HK, CAPS); self.assertTrue(ok, msg); self.assertIn('step_mult %d' % k, msg)
        info, _ = self.ok(raw({'step_mult': 4, 'lr': 1e-06, 'kl': 0.005}))
        self.assertEqual(info['train'], dict(lr=1e-06, kl=0.005, step_mult=4, **F.KNOB_DEFAULTS)); self.assertEqual(info['train_given'], ['kl', 'lr', 'step_mult'])
    def test_step_mult_bad(self):
        for v in (0, 9, -1, 16, 2.0, 1.5, '2', True, None, [2]):
            self.bad(raw({'step_mult': v}), 'step_mult')
        self.bad(raw({'steps': 2}), 'train keys')
        self.bad(raw({'step_mult': 2}, version=F.FORMAT), 'top level keys')
    def test_knobs(self):
        for k, (kind, b, d) in F.KNOB_SPEC.items():
            vals = [b[0], b[1], (b[0] + b[1]) / 2] if kind == 'number' else list(b)
            for v in vals:
                info, _ = self.ok(raw({k: v}))
                self.assertEqual(info['train'][k], float(v) if kind == 'number' else v); self.assertEqual(info['train_given'], [k])
                ok, msg = AB.check(raw({k: v}), HK, CAPS); self.assertTrue(ok, msg)
            bads = [b[0] - abs(b[0]) - 1e-12, b[1] * 2 + 1, '1', True, None] if kind == 'number' else ['x', 1, None, True, b[0].upper()]
            for v in bads:
                self.bad(raw({k: v}), k)
        info, _ = self.ok(raw(dict(F.KNOB_DEFAULTS, adv_clip=1.0)))
        self.assertEqual({k: info['train'][k] for k in F.KNOB_SPEC}, dict(F.KNOB_DEFAULTS, adv_clip=1.0))
    def test_v2_disabled(self):
        ok, info, _ = F.check(raw({'step_mult': 2}), HK, TOK, 125, 1000, False); self.assertFalse(ok)
if __name__ == '__main__':
    unittest.main(verbosity=1)
