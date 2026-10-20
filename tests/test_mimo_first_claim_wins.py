"""First-claim-wins (earliest Cloudflare R2 LastModified) for MiMo epochs."""
import unittest
from unittest.mock import patch

import test_mimo_coordinator as coordinator_fixtures
from subnet.claim_order import RULE, resolve, contested, winners, tie_break, enabled
from subnet.complete_group_capture import paths


class ClaimOrderUnitTests(unittest.TestCase):
    def test_earliest_wins_and_records_contest(self):
        r = resolve([('a', ('math', 1), 20.0), ('b', ('math', 1), 10.0), ('c', ('math', 2), 5.0)])
        self.assertEqual(r[('math', 1)], ('b', 10.0, 2))
        self.assertEqual(r[('math', 2)], ('c', 5.0, 1))
        self.assertEqual(contested(r), [dict(key=['math', 1], winner='b', received_at=10.0, claimants=2)])
        self.assertEqual(winners(contested(r)), {('math', 1): 'b'})

    def test_same_second_tie_break_is_order_independent(self):
        key = ('math', 3); claims = [(m, key, 100.0) for m in ('m1', 'm2', 'm3')]
        expected = min(('m1', 'm2', 'm3'), key=lambda m: tie_break(m, key))
        self.assertEqual(resolve(claims)[key][0], expected)
        self.assertEqual(resolve(list(reversed(claims)))[key][0], expected)

    def test_timestamped_beats_untimestamped_and_none_is_zero_for_all(self):
        self.assertEqual(resolve([('a', ('e', 1), None), ('b', ('e', 1), 9e9)])[('e', 1)][0], 'b')
        self.assertEqual(resolve([('a', ('e', 1), None), ('b', ('e', 1), None)])[('e', 1)], (None, None, 2))

    def test_rule_gate(self):
        self.assertFalse(enabled({}))
        self.assertTrue(enabled({'claim_order_rule': RULE}))
        with self.assertRaises(ValueError):
            enabled({'claim_order_rule': 'something-else'})

    def test_snapshot_counts_contested_winner_only(self):
        from subnet.continuous_audit_policy import snapshot, VERSION
        h = lambda c: c * 64
        row = lambda miner: dict(epoch='e', round=1, checkpoint=h('c'), miner=miner, env_id='math', index=1,
            batch_sha256=h(miner[0]), proof_sha256=h('f'), commitment_sha256=h('d'), verifier_contract_sha256=h('e'),
            committed_at=10.0)
        rows = [row(h('a')), row(h('b'))]
        policy = dict(version=VERSION, recent_epochs=4, decay=1, prior_alpha=1, prior_beta=1, invalid_multiplier=.5,
                      zero_epoch_after=0, blacklist_after=0, blacklist_epochs=0)
        args = dict(epoch='e', round=1, checkpoint=h('c'), cutoff=100.0, audit_policy=policy)
        legacy = snapshot(rows, [], {}, **args)
        self.assertEqual(legacy['points'], {h('a'): 0.0, h('b'): 0.0})
        self.assertNotIn('claim_order_rule', legacy)
        order = dict(claim_order_rule=RULE, contested_claims=[dict(key=['math', 1], winner=h('b'), received_at=5.0, claimants=2)])
        ordered = snapshot(rows, [], {}, claim_order=order, **args)
        self.assertEqual(ordered['points'][h('a')], 0.0)
        self.assertGreater(ordered['points'][h('b')], 0.0)
        self.assertEqual(ordered['claim_order_rule'], RULE)
        self.assertEqual(ordered['contested_claims'], order['contested_claims'])


class CoordinatorFirstClaimTests(unittest.TestCase):
    def setUp(self):
        original = coordinator_fixtures.fixture
        def ordered_fixture():
            m = original(); m['claim_order_rule'] = RULE; return m
        self.f = coordinator_fixtures.CoordinatorIntegrationTests('test_cross_miner_collisions_zero_both_and_keep_unique_task')
        with patch('test_mimo_coordinator.fixture', side_effect=ordered_fixture):
            self.f.setUp()
        self.addCleanup(self.f.doCleanups)

    def put(self, actor, index):
        f = self.f; runtime = coordinator_fixtures.FixtureRuntime(f.opened, actor.identity.id)
        def mutate(runtime, row):
            row['turns'][0]['text'] = f.tokenizer.decode(row['turns'][0]['output'])
        runtime.mutate = mutate
        with patch('subnet.miner.make_runtime', return_value=runtime):
            actor.search(index, env_id='math')
        self.assertEqual(actor.upload(), 200)

    def collect(self, actor):
        return self.f.coordinator.collect(actor.identity.id, now=self.f.bucket.clock + 1)

    def test_capture_records_r2_times_and_earliest_upload_wins(self):
        f = self.f; first = f.actor(f.miners[0]); second = f.actor(f.miners[1])
        self.put(first, 0); captured = self.collect(first)['payload']
        self.put(second, 0); self.put(second, 1); self.collect(second)
        self.assertIn('received_at', captured)
        self.assertTrue(all(type(d['received_at']) is float for d in captured['documents']))
        dispatch = f.seal()
        self.assertEqual(sorted((r['miner'], r['index']) for r in dispatch['selected']),
                         sorted([(first.identity.id, 0), (second.identity.id, 1)]))
        self.assertEqual([(r['miner'], r['reason']) for r in dispatch['excluded']],
                         [(second.identity.id, 'cross-miner-task-collision')])
        self.assertEqual(dispatch['claim_order_rule'], RULE)
        (claim,) = dispatch['contested_claims']
        self.assertEqual((claim['key'], claim['winner'], claim['claimants']), (['math', 0], first.identity.id, 2))

    def test_same_second_upload_tie_break(self):
        f = self.f; actors = [f.actor(f.miners[0]), f.actor(f.miners[1])]
        for actor in actors: self.put(actor, 0)
        keys = [paths(f.opened, a.identity.id, 0)[0] for a in actors]
        stamp = min(f.bucket.objects[k][1] for k in keys)
        for k in keys: f.bucket.objects[k] = (f.bucket.objects[k][0], stamp)
        for actor in actors: self.collect(actor)
        dispatch = f.seal()
        expected = min((a.identity.id for a in actors), key=lambda m: tie_break(m, ('math', 0)))
        self.assertEqual([r['miner'] for r in dispatch['selected']], [expected])
        self.assertAlmostEqual(dispatch["contested_claims"][0]["received_at"], stamp, places=5)


if __name__ == '__main__':
    unittest.main()
