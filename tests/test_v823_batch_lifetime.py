"""v8.2.3 batch lifetime (pt765u): replace until seal and head-bound draws."""
from test_complete_group_transport import *
from test_complete_group_transport import TransportTests
import copy as _copy
import sqlite3
from subnet import batch_lifetime as BL
from subnet.complete_group_mining import replace, carry
from subnet.complete_group_sampling import prepare_draws


class BatchLifetimeTests(unittest.TestCase):
    setUp = TransportTests.setUp
    actor = TransportTests.actor
    search = TransportTests.search
    resolve = TransportTests.resolve
    capture = TransportTests.capture

    def lm(self, slots=None, base=None, epoch=None, checkpoint=None, randomness=None):
        m = fixture(8) if base is None else _copy.deepcopy(base)
        m.pop('capabilities', None)
        m[BL.POLICY_KEY] = dict(version=BL.VERSION, upload_slots=slots or 2 * m['max_batches'])
        if epoch: m['epoch'] = epoch
        if checkpoint: m['checkpoint']['id'] = checkpoint
        if randomness: m['sampling_contract']['randomness'] = randomness
        m['capabilities'] = {self.identity.id: capability(self.bucket, m, self.identity.id, registered_miners={self.identity.id})}
        return m

    def commitment(self, m):
        return loads(self.bucket.objects[paths(m, self.identity.id)][0], 10**7)['payload']

    def supersessions(self):
        with sqlite3.connect(self.ledger.path) as db:
            return db.execute('SELECT sequence,groups FROM supersessions ORDER BY sequence').fetchall()

    # ---- replace until seal
    def test_replace_before_seal_takes_effect(self):
        m = self.lm(); a = self.actor(m, self.state)
        self.search(a, 0); self.search(a, 1); a.upload()
        self.assertEqual(self.capture(m)['total_groups'], 2)
        replace(a, lambda b: b['index'] != 0)
        result = self.capture(m)
        self.assertEqual(result['total_groups'], 1)
        self.assertEqual([g['index'] for g in self.commitment(m)['groups']], [1])
        self.search(a, 2); a.upload()
        result = self.capture(m)
        claims = self.commitment(m)['groups']
        self.assertEqual(result['total_groups'], 2)
        self.assertEqual([(g['index'], g['slot']) for g in claims], [(1, 1), (2, 2)])  # kept slot, new write-once slot
        self.assertEqual(self.supersessions(), [(0, 2), (1, 1), (2, 2)])  # auditable, hash-chained history
        # The dropped task cannot be re-sampled into credit (draws are fixed).
        with self.assertRaises(RuntimeError): self.search(a, 0)

    def test_replace_after_deadline_is_rejected_and_sealed_set_stands(self):
        m = self.lm(); a = self.actor(m, self.state)
        self.search(a, 0); self.search(a, 1); a.upload(); first = self.capture(m)
        replace(a, lambda b: b['index'] != 0)
        key = paths(m, self.identity.id); body, _ = self.bucket.objects[key]
        self.bucket.objects[key] = (body, m['deadline'])  # landed at/after the signed deadline
        with self.assertRaises(ValueError): self.capture(m)
        self.assertEqual(self.supersessions(), [(0, 2)])  # the pre-deadline commitment is still the latest
        with patch('time.time', return_value=m['deadline'] + 1):
            from subnet.miner import EpochClosed
            with self.assertRaises(EpochClosed): replace(a, lambda b: False)
        self.assertEqual(first['total_groups'], 2)

    def test_older_commitment_cannot_supersede_newer(self):
        m = self.lm(); a = self.actor(m, self.state)
        self.search(a, 0); a.upload(); self.capture(m)
        key = paths(m, self.identity.id); old_body, old_stamp = self.bucket.objects[key]
        self.search(a, 1); a.upload(); self.capture(m)
        self.assertEqual(self.capture(m)['new_groups'], 0)  # re-capturing the latest is idempotent
        self.bucket.objects[key] = (old_body, old_stamp - 5)
        with self.assertRaises(ValueError): self.capture(m)

    def test_withdraw_and_slot_bound(self):
        m = self.lm(slots=3); a = self.actor(m, self.state)
        self.search(a, 0); self.search(a, 1); self.search(a, 2); a.upload(); self.capture(m)
        replace(a, lambda b: False)
        self.assertEqual(self.capture(m)['total_groups'], 0)
        self.search(a, 3)
        with self.assertRaisesRegex(ValueError, 'write-once upload slots'): a.upload()  # bounded storage per hotkey/epoch
        self.assertEqual(self.commitment(m)['groups'], [])  # previous (empty) commitment stays in force

    def test_slots_must_be_distinct_ascending_and_bounded(self):
        m = self.lm(); a = self.actor(m); self.search(a, 0); self.search(a, 1)
        for slots in ([1, 0], [0, 0], [0, 6]):
            with self.assertRaises(ValueError): signed_commitment(self.identity, m, a._group_prepared, slots)
        signed_commitment(self.identity, m, a._group_prepared, [2, 5])

    def test_legacy_manifest_keeps_cumulative_contiguous_rules(self):
        m = fixture(8); m['capabilities'] = {self.identity.id: capability(self.bucket, m, self.identity.id, registered_miners={self.identity.id})}
        a = self.actor(m); self.search(a, 0); self.search(a, 1)
        with self.assertRaises(ValueError): signed_commitment(self.identity, m, a._group_prepared, [0, 2])
        with self.assertRaises(ValueError): replace(a, lambda b: False)

    # ---- head-bound draws
    def contexts(self, *manifests):
        from subnet.forced_sampling import binding
        return [binding(x, self.identity.id) for x in manifests]

    def test_draws_are_head_bound_seed_fixed_and_epoch_independent(self):
        m1 = self.lm(); m2 = self.lm(base=m1, epoch=m1['epoch'] + '-next')
        green = self.lm(base=m1, epoch=m1['epoch'] + '-green', checkpoint='e' * 64, randomness='d' * 64)
        c1, c2, c3 = self.contexts(m1, m2, green)
        args = ('math', digest(dict(index=0, taskset='taskset-1')), 0, 3, 0, [1, 2])
        d1, d2, d3 = (prepare_draws(c, *args) for c in (c1, c2, c3))
        self.assertEqual([d1(i) for i in range(5)], [d2(i) for i in range(5)])
        self.assertNotEqual([d1(i) for i in range(5)], [d3(i) for i in range(5)])
        legacy1 = fixture(8); legacy2 = _copy.deepcopy(legacy1); legacy2['epoch'] += '-next'
        l1, l2 = self.contexts(legacy1, legacy2)
        self.assertNotEqual(prepare_draws(l1, *args)(0), prepare_draws(l2, *args)(0))  # old semantics untouched

    def test_head_bound_batch_carries_over_on_unchanged_head(self):
        m1 = self.lm(); a1 = self.actor(m1, self.state); self.search(a1, 0); self.search(a1, 1); a1.upload()
        m2 = self.lm(base=m1, epoch=m1['epoch'] + '-next')
        state2 = Path(self.tmp.name) / 'miner2.json'; a2 = self.actor(m2, state2)
        self.assertEqual(carry(a2, m1, self.state), 2)
        self.assertTrue(all(b['epoch'] == m2['epoch'] for b, _ in a2._group_prepared))
        self.assertEqual(a2.upload(), 200)
        self.assertEqual(self.capture(m2)['total_groups'], 2)
        # A restart restores the carried state without the original attempts.
        a2._group_journal.close(); a3 = self.actor(m2, state2); self.assertEqual(len(a3.batches), 2)

    def test_head_bound_batch_rejected_after_green(self):
        m1 = self.lm(); a1 = self.actor(m1, self.state); self.search(a1, 0); a1.upload()
        green = self.lm(base=m1, epoch=m1['epoch'] + '-green', checkpoint='e' * 64, randomness='d' * 64)
        a2 = self.actor(green, Path(self.tmp.name) / 'g.json')
        self.assertEqual(carry(a2, m1, self.state), 0)
        self.assertFalse(BL.carries(m1, green))
        c1, cg = self.contexts(m1, green)
        with self.assertRaisesRegex(ValueError, 'resample'): BL.restamp(a1._group_prepared[0][0], c1, cg)
        # Forcing the old batch into the new epoch fails the structural binding.
        moved = _copy.deepcopy(a1._group_prepared[0][0]); moved['epoch'] = green['epoch']
        moved['sampling_context_sha256'] = digest(cg); moved['checkpoint'] = green['checkpoint']['id']
        with self.assertRaises(ValueError):
            from subnet.complete_group_mining import validate_group
            validate_group(moved, green, a2.batch_contract, cg, self.identity.id)

    def test_host_seed_is_fixed_per_head_and_fresh_after_head_change(self):
        a = BL.carry(None, 'a' * 64, 'e1', '1' * 64)
        b = BL.carry(a, 'a' * 64, 'e2', '2' * 64)
        c = BL.carry(b, 'c' * 64, 'e3', '3' * 64)
        self.assertEqual((b['randomness'], b['origin_epoch']), ('1' * 64, 'e1'))
        self.assertEqual((c['randomness'], c['origin_epoch']), ('3' * 64, 'e3'))


if __name__ == '__main__':
    unittest.main()
