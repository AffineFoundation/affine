"""CPU test: synthetic affine-batch-v1 file -> lifecycle tokenization/documents -> signed FF dispatch -> trainer host v8.3.3h
ff_trainer.prepare -> v8211_batched._layout (trainer input accepted); and native groups give identical rows old vs new."""
import hashlib, importlib.util, json, random, sys, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
RA = Path('/home/const/subnet120-rewrite/state/root-audits')
D = RA / '20261030-free-file'
SRC = RA / '20261029-v829u-125task/deployment-final-source-v823u'
OLD = RA / '20261030-trainer-speed/release-v833g-signed/trainer-host-v833g'
NEW = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else D / 'stage/host-v833h'
sys.argv = sys.argv[:1]
sys.path[:0] = [str(SRC), str(D / 'release'), str(NEW)]
import torch
from subnet.storage import Identity
from subnet.mimo_release_policy import sign
from subnet.mimo_grpo.config import TrainerConfig
from subnet.mimo_grpo.data import TaskGroup, Trajectory, digest
import ff_format as F
import ff_lifecycle as FL


def load(name, path):
    s = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(s); sys.modules[name] = m
    s.loader.exec_module(m); return m


OLDB, NEWB = load('b_old', OLD / 'v8211_batched.py'), load('b_new', NEW / 'v8211_batched.py')
FT = load('ff_trainer', NEW / 'ff_trainer.py')
LIFE = Path('/home/const/subnet120-rewrite/state/mimo-mainnet-v1')
CUR = json.loads((LIFE / 'CURRENT.json').read_bytes())
E = CUR['epoch_index']
OPEN = json.loads((LIFE / 'epochs' / str(E) / 'coordinator' / 'OPENING.json').read_bytes())['payload']
JOB = next(json.loads(p.read_bytes()) for p in sorted((LIFE / 'epochs').glob('*/TRAINING-JOB.json'), key=lambda p: -int(p.parent.name)))
CFG = TrainerConfig(**JOB['payload']['training']['config']['settings'])


def rows_key(rows):
    return [(r['advantage'], r['reward'], r['active'], r['keep'], r['rollout'].nonce, r['group'].task_id) for r in rows]


class Trainer(unittest.TestCase):
    def test_native_rows_identical(self):
        rnd = random.Random(7); eos = OPEN['mimo_runtime']['eos_token_ids'][0]
        groups = []
        for t in range(20):
            outs = ['correct', 'wrong'] + [rnd.choice(['correct', 'wrong', 'unresolved']) for _ in range(6)]
            trs = []
            for n, o in enumerate(outs):
                L = rnd.randint(5, 9000)
                stop = 'max_tokens' if o == 'unresolved' and rnd.random() < .5 else 'eos'
                toks = tuple(rnd.randrange(1000) for _ in range(L - 1)) + ((eos,) if stop == 'eos' else (7,))
                trs.append(Trajectory(n, toks, o, stop))
            groups.append(TaskGroup('task-%d' % t, (1, 2, 3), tuple(trs), 'a' * 64))
        self.assertEqual(rows_key(OLDB._layout(groups, CFG)), rows_key(NEWB._layout(groups, CFG)))
        self.assertEqual(rows_key(OLDB.layout_drop_unresolved(groups, CFG)[0]), rows_key(NEWB.layout_drop_unresolved(groups, CFG)[0]))

    def test_ff_end_to_end(self):
        tok = F.tokenizer('/home/const/subnet120-rewrite/state/root-audits/20261030-determinism/tok',
                          {n: CUR['checkpoint']['files'][n] for n in FL.TOKENIZER_FILES})
        hk = '5FakeHotkey'
        groups = [dict(prompt='Compute %d*%d.' % (i, i + 1), rollouts=[dict(output='<think>\nx</think>\n\n%d' % (i * (i + 1) + k), reward=float(k == 0))
                                                                    for k in range(1 + i % 5)]) for i in range(12)]
        groups.append(dict(prompt='long one', rollouts=[dict(output='a' + ' a' * 8191, reward=-0.5), dict(output='b', reward=2.0)]))
        raw = json.dumps(dict(version=F.FORMAT, hotkey=hk, groups=groups)).encode()
        v = F.validate(raw, hk, tok, 125, 1000)
        ident = Identity()
        m = dict(OPEN)
        turn = dict(hotkey=hk, groups=v['groups'])
        docs = FL.documents(turn, m, CUR)
        with tempfile.TemporaryDirectory() as td:
            for h, b, _ in docs:
                Path(td, h + '.json').write_bytes(b)
            rows = [x[2] for x in docs]
            payload = dict(version=FT.VERSION, epoch=m['epoch'], run_id=m['run_id'], checkpoint=CUR['checkpoint']['id'],
                           manifest_sha256=digest(m), training_requires_sampling_audit=False, hotkey=hk, selected=rows,
                           selected_task_count=len(rows), selected_rollout_count=sum(r['rollouts'] for r in rows))
            opening, dispatch = sign(ident, m), sign(ident, payload)
            self.assertTrue(FT.is_ff(dispatch))
            shell = SimpleNamespace(checkpoint_id=CUR['checkpoint']['id'], spec=SimpleNamespace(to_dict=lambda: m['mimo_runtime']),
                                    vocab_size=m['group_transport_policy']['vocab_size'], eos_token_ids=tuple(m['mimo_runtime']['eos_token_ids']))
            self.assertEqual(m['checkpoint']['id'], CUR['checkpoint']['id'])
            p = FT.prepare(opening, dispatch, authority=ident.id, documents_root=td, runtime=shell, optimizer_step=5)
            gs = p['groups']
            self.assertEqual(len(gs), 13); self.assertEqual([len(g.trajectories) for g in gs][:5], [1, 2, 3, 4, 5])
            for g in gs:   # the batched engine's own check (size argument ignored for free groups)
                g.validate(CFG.group_size, vocab_size=shell.vocab_size, eos_token_ids=set(shell.eos_token_ids),
                           unresolved_failures=True)
            capped = gs[-1].trajectories[0]
            self.assertEqual((capped.stop, capped.outcome, len(capped.tokens)), ('max_tokens', 'unresolved', 8192))
            rows_ = NEWB._layout(gs, CFG)
            self.assertEqual(len(rows_), sum(len(g.trajectories) for g in gs))
            pen = lambda n: CFG.reward('wrong', n)
            for g in gs:
                rs = [r for r in rows_ if r['group'] is g]
                rew = [t.reward + pen(len(t.tokens)) for t in g.trajectories]
                self.assertEqual([r['reward'] for r in rs], rew)
                mean = torch.tensor(rew, dtype=torch.float32).mean().item() if len(rew) > 1 else 0.0
                for r, x in zip(rs, rew):
                    self.assertAlmostEqual(r['advantage'], x - mean, places=5)
            # report/receipt shape the worker expects
            self.assertEqual(set(p), {'manifest', 'dispatch', 'binding', 'groups', 'receipts', 'diagnostic_omitted_groups'})
            from subnet.mimo_grpo.service import training_job
            training_job(gs, p['binding'], CFG, manifest_sha256=digest(m), source_closure_sha256='0' * 64)
            # tampering is refused
            bad = sign(ident, dict(payload, hotkey='someone-else'))
            with self.assertRaises(ValueError):
                FT.prepare(opening, bad, authority=ident.id, documents_root=td, runtime=shell)
            import pickle
            self.assertEqual(pickle.loads(pickle.dumps(gs)), gs)   # DP helpers receive groups pickled
        print(json.dumps(dict(groups=len(gs), rollouts=len(rows_), cfg=dict(group_size=CFG.group_size, cap_policy=CFG.cap_policy,
                                                                          overlong=CFG.overlong_penalty))))


if __name__ == '__main__':
    unittest.main(verbosity=1)
