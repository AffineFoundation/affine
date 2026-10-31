"""Unit tests: ff_format affine-batch-v2 (optional train params) + v1 unchanged. RA/20261031-ff-train-params. Arg: dir holding ff_format.py."""
import json, sys, unittest
sys.path.insert(0, sys.argv.pop(1))
import ff_format as F

TOK = F.tokenizer('/home/const/subnet120-rewrite/state/root-audits/20261030-determinism/tok')
HK = '5C4khFWyExampleHotkeyxxxxxxxxxxxxxxxxxxxxxxxxxx'
G = [dict(prompt='What is 2+2?', rollouts=[dict(output='4', reward=1.0), dict(output='5', reward=0)])]


def raw(**kw):
    return json.dumps(dict(dict(version=F.FORMAT_V2, hotkey=HK, groups=G), **kw)).encode()


class V2(unittest.TestCase):
    def ok(self, b, v2=True):
        ok, info, groups = F.check(b, HK, TOK, 125, 1000, v2)
        self.assertTrue(ok, info); return info, groups

    def bad(self, b, frag, v2=True):
        ok, info, _ = F.check(b, HK, TOK, 125, 1000, v2)
        self.assertFalse(ok); self.assertIn(frag, info); return info

    def test_v1_unchanged(self):
        b = json.dumps(dict(version=F.FORMAT, hotkey=HK, groups=G)).encode()
        for v2 in (False, True):
            info, groups = self.ok(b, v2)
            self.assertEqual((info['format'], info['train'], info['train_given']), (F.FORMAT, dict(lr=5e-07, kl=0.02), []))
        v1 = F.validate(b, HK, TOK, 125, 1000, False)
        self.assertEqual(v1['groups'], F.validate(b, HK, TOK, 125, 1000, True)['groups'])
        self.assertIn('top level keys must be exactly version, hotkey, groups', self.bad(json.dumps(dict(version=F.FORMAT, hotkey=HK, groups=G, train={})).encode(), 'train" needs affine-batch-v2'))
        self.assertEqual(self.bad(json.dumps(dict(version=F.FORMAT, hotkey=HK, groups=G, train={})).encode(), 'top level', v2=False),
                         'top level keys must be exactly version, hotkey, groups')

    def test_v2_params(self):
        info, g2 = self.ok(raw(train=dict(lr=1e-6, kl=0.005)))
        self.assertEqual((info['format'], info['train'], info['train_given']), (F.FORMAT_V2, dict(lr=1e-06, kl=0.005), ['kl', 'lr']))
        info, _ = self.ok(raw(train=dict(lr=1.25e-7)));  self.assertEqual((info['train'], info['train_given']), (dict(lr=1.25e-07, kl=0.02), ['lr']))
        info, _ = self.ok(raw(train=dict(kl=0.1)));      self.assertEqual((info['train'], info['train_given']), (dict(lr=5e-07, kl=0.1), ['kl']))
        info, _ = self.ok(raw(train={}));                self.assertEqual((info['train'], info['train_given']), (dict(lr=5e-07, kl=0.02), []))
        info, g0 = self.ok(raw());                       self.assertEqual(info['train'], dict(lr=5e-07, kl=0.02))
        self.assertEqual(g0, g2)                         # params never change the tokenized groups
        self.assertIs(type(self.ok(raw(train=dict(lr=1e-6, kl=0.05)))[0]['train']['kl']), float)

    def test_v2_invalid(self):
        self.bad(raw(train=dict(lr=1.24e-7)), 'train.lr must be a number in [1.25e-07, 1e-06]')
        self.bad(raw(train=dict(lr=1.01e-6)), 'train.lr must be a number')
        self.bad(raw(train=dict(lr=0.001)), 'train.lr must be')
        self.bad(raw(train=dict(kl=0.004)), 'train.kl must be a number in [0.005, 0.1]')
        self.bad(raw(train=dict(kl=0.5)), 'train.kl must be')
        self.bad(raw(train=dict(kl=0)), 'train.kl must be')
        self.bad(raw(train=dict(lr='5e-7')), 'train.lr must be')
        self.bad(raw(train=dict(lr=True)), 'train.lr must be')
        self.bad(raw(train=dict(lr=None)), 'train.lr must be')
        self.bad(raw(train=dict(lr=5e-7, momentum=0.9)), 'train keys must be among lr, kl')
        self.bad(raw(train=[5e-7]), 'train must be an object')
        self.bad(raw(train=None), 'train must be an object')
        self.bad(raw(extra=1), 'optionally train')
        self.bad(b'{"version":"affine-batch-v2","hotkey":"%s","groups":[{"prompt":"p","rollouts":[{"output":"a","reward":1}]}],"train":{"lr":NaN}}' % HK.encode(), 'non-finite')
        self.bad(b'{"version":"affine-batch-v2","hotkey":"%s","groups":[{"prompt":"p","rollouts":[{"output":"a","reward":1}]}],"train":{"lr":1e999}}' % HK.encode(), 'train.lr must be')

    def test_v2_disabled(self):
        self.assertEqual(self.bad(raw(train=dict(lr=1e-6)), 'top level', v2=False), 'top level keys must be exactly version, hotkey, groups')
        self.assertEqual(self.bad(raw(), 'version', v2=False), 'version must be affine-batch-v1')


if __name__ == '__main__':
    unittest.main(verbosity=1)
