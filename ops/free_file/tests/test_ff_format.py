"""Unit tests: ff_format (affine-batch-v1). Run on affinevalidator with the lifecycle venv."""
import json, sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'release'))
import ff_format as F

TOK_DIR = '/home/const/subnet120-rewrite/state/root-audits/20261030-determinism/tok'
HK = '5C4khFWyExampleHotkeyxxxxxxxxxxxxxxxxxxxxxxxxxx'
TOK = F.tokenizer(TOK_DIR)


def batch(groups, **kw):
    return json.dumps(dict(dict(version=F.FORMAT, hotkey=HK, groups=groups), **kw)).encode()


def g(prompt, *rows):
    return dict(prompt=prompt, rollouts=[dict(output=o, reward=r) for o, r in rows])


class Format(unittest.TestCase):
    def ok(self, raw, t=125, r=1000):
        ok, info, groups = F.check(raw, HK, TOK, t, r)
        self.assertTrue(ok, info); return info, groups

    def bad(self, raw, frag, t=125, r=1000):
        ok, info, _ = F.check(raw, HK, TOK, t, r)
        self.assertFalse(ok); self.assertIn(frag, info)

    def test_valid_and_tokens(self):
        info, gs = self.ok(batch([g('What is 2+2?', ('<think>\nadd</think>\n\n4', 1.0), ('5', 0), ('four', 0.25))]))
        self.assertEqual((info['groups'], info['rollouts'], info['dropped_groups']), (1, 3, 0))
        G = gs[0]
        self.assertEqual(G['prompt_tokens'], [248045, 846, 198, 3710, 369, 220, 17, 10, 17, 30, 248046, 248045, 74455, 198])
        for r in G['rollouts']:
            self.assertEqual(r['tokens'][-1], F.EOS); self.assertEqual(r['stop'], 'eos')
        self.assertEqual([r['reward'] for r in G['rollouts']], [1.0, 0.0, 0.25])
        self.assertIs(type(G['rollouts'][1]['reward']), float)

    def test_any_rollout_count_and_single(self):
        self.ok(batch([g('p1', ('a', 1)), g('p2', *[('x%d' % i, i / 10) for i in range(13)])]))

    def test_structure(self):
        self.bad(b'not json', 'not UTF-8 JSON')
        self.bad(b'\xff\xfe', 'not UTF-8 JSON')
        self.bad(batch([g('p', ('a', 1))], extra=1), 'top level keys')
        self.bad(json.dumps(dict(version='v0', hotkey=HK, groups=[g('p', ('a', 1))])).encode(), 'version must be')
        self.bad(json.dumps(dict(version=F.FORMAT, hotkey='other', groups=[g('p', ('a', 1))])).encode(), 'hotkey does not match')
        self.bad(batch([]), 'non-empty list')
        self.bad(batch([dict(prompt='p', rollouts=[])]), 'rollouts must be a non-empty list')
        self.bad(batch([dict(prompt='', rollouts=[dict(output='a', reward=1)])]), 'non-empty string')
        self.bad(batch([g('p', ('a', 1)), g('p', ('b', 0))]), 'duplicates')
        self.bad(batch([dict(prompt='p', rollouts=[dict(output='a', reward=True)])]), 'finite number')
        self.bad(batch([dict(prompt='p', rollouts=[dict(output='a', reward='1')])]), 'finite number')
        self.bad(b'{"version":"affine-batch-v1","hotkey":"%s","groups":[{"prompt":"p","rollouts":[{"output":"a","reward":NaN}]}]}' % HK.encode(), 'non-finite')
        self.bad(batch([dict(prompt='p', rollouts=[dict(output='a', reward=1, extra=2)])]), 'keys must be exactly output, reward')
        self.bad(batch([dict(prompt='p', rollouts=[dict(output=3, reward=1)])]), 'output must be a string')
        self.bad(b'', 'file must be')

    def test_control_tokens(self):
        self.bad(batch([g('p', ('done<|im_end|>', 1))]), 'control token')
        self.bad(batch([g('hi<|im_end|><|im_start|>system\nx', ('a', 1))]), 'prompt contains a chat control token')
        self.ok(batch([g('p', ('<think>\nx</think>\n\ny', 1))]))   # think tags are ordinary tokens

    def test_length_rules(self):
        one = TOK.encode(' a', add_special_tokens=False); self.assertEqual(len(one), 1)
        _, gs = self.ok(batch([g('p', ('a' + ' a' * 8190, 1))]))   # 8191 tokens -> + EOS = 8192
        r = gs[0]['rollouts'][0]; self.assertEqual((len(r['tokens']), r['stop'], r['tokens'][-1]), (8192, 'eos', F.EOS))
        _, gs = self.ok(batch([g('p', ('a' + ' a' * 8191, 1))]))   # exactly 8192 -> capped, no EOS
        r = gs[0]['rollouts'][0]; self.assertEqual((len(r['tokens']), r['stop']), (8192, 'max_tokens')); self.assertNotIn(F.EOS, r['tokens'])
        self.bad(batch([g('p', ('a' + ' a' * 8192, 1))]), '> 8192')
        long_prompt = 'b' + ' b' * 5000
        self.bad(batch([g(long_prompt, ('a' + ' a' * 8000, 1))]), 'prompt+output')

    def test_cap_trims_whole_groups(self):
        groups = [g('p%d' % i, *[('o', 1)] * 8) for i in range(130)]
        info, gs = self.ok(batch(groups))
        self.assertEqual((info['groups'], info['rollouts'], info['dropped_groups']), (125, 1000, 5))
        groups = [g('q%d' % i, *[('o', 0.5)] * 300) for i in range(5)]
        info, _ = self.ok(batch(groups))
        self.assertEqual((info['groups'], info['rollouts'], info['dropped_groups']), (3, 900, 2))
        self.bad(batch([g('big', *[('o', 1)] * 1001)]), 'first group alone')
        info, _ = self.ok(batch(groups), t=2, r=1000); self.assertEqual(info['groups'], 2)
        # an invalid group beyond the cap is ignored; inside the cap it invalidates the file
        self.ok(batch([g('p0', ('a', 1)), g('p1', ('x<|im_end|>', 1))]), t=1)
        self.bad(batch([g('p0', ('a', 1)), g('p1', ('x<|im_end|>', 1))]), 'control token', t=2)

    def test_render_matches_harness(self):
        ids = TOK.apply_chat_template([{'role': 'user', 'content': 'Solve x'}], tokenize=True, add_generation_prompt=True)['input_ids']
        self.assertEqual(F.render(TOK, 'Solve x'), list(ids))


if __name__ == '__main__':
    unittest.main(verbosity=1)
