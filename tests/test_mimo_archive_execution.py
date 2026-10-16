import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ops.mimo_lifecycle import archive_execution_resolver

V7, V82, V821 = 'a7' * 32, 'b8' * 32, 'c9' * 32
INV = {a: dict(source_files={'subnet/x.py': a[:8], 'subnet/n.py': str(n)}, runtime_versions={'torch': '2'})
       for n, a in enumerate((V7, V82, V821))}


class Policy:
    def __init__(self, sha, archive):
        self.policy_sha256 = sha; self.envelope = {'p': sha}
        self.scope = {'immutable_manifest': {'source_bundle': {'sha256': archive}}}

    def validate_manifest(self, m):
        if m['policy'] != self.envelope: raise ValueError('policy')
        return m


class Chain:
    def __init__(self, *policies):
        self.members = [(i, p) for i, p in enumerate(policies)]
        self.active = policies[-1]

    @property
    def policy_sha256(self): return self.active.policy_sha256

    def policy_for_manifest(self, m):
        for _, p in self.members:
            if m['policy'] == p.envelope: p.validate_manifest(m); return p
        raise ValueError('not a chain member')

    def validate_manifest(self, m): self.policy_for_manifest(m); return m


P7, P82, P821 = Policy('7' * 64, V7), Policy('8' * 64, V82), Policy('9' * 64, V821)
manifest = lambda p, archive=None: {'policy': p.envelope, 'epoch': 1,
                                    'source_bundle': {'sha256': archive or p.scope['immutable_manifest']['source_bundle']['sha256']}}


class ArchiveExecutionTests(unittest.TestCase):
    def chain(self): return Chain(P7, P82, P821)

    def test_v7_epoch_gets_v7_inventory_not_live(self):
        r = archive_execution_resolver(self.chain(), {V7: INV[V7], V82: INV[V82], V821: INV[V821]}, INV[V821])
        self.assertEqual(r(manifest(P7)), INV[V7]); self.assertNotEqual(r(manifest(P7)), INV[V821])

    def test_v82_epoch_gets_v82_inventory(self):
        r = archive_execution_resolver(self.chain(), {V7: INV[V7], V82: INV[V82], V821: INV[V821]}, INV[V821])
        self.assertEqual(r(manifest(P82)), INV[V82])

    def test_mixed_epochs_each_get_own_archive(self):
        r = archive_execution_resolver(self.chain(), {V7: INV[V7], V82: INV[V82], V821: INV[V821]}, INV[V821])
        got = [r(manifest(p))['source_files'] for p in (P7, P821, P82, P7, P821)]
        self.assertEqual(got, [INV[a]['source_files'] for a in (V7, V821, V82, V7, V821)])

    def test_v82_chain_reproduces_the_live_defect_fix(self):
        # live v8.2 deployment; v7 epoch must still get the 770-file v7 inventory
        r = archive_execution_resolver(Chain(P7, P82), {V7: INV[V7], V82: INV[V82]}, INV[V82])
        self.assertEqual(r(manifest(P7)), INV[V7]); self.assertEqual(r(manifest(P82)), INV[V82])

    def test_result_is_a_copy(self):
        r = archive_execution_resolver(self.chain(), {V7: INV[V7], V82: INV[V82], V821: INV[V821]}, INV[V821])
        r(manifest(P7))['source_files']['subnet/x.py'] = 'tamper'
        self.assertEqual(r(manifest(P7)), INV[V7])

    def test_manifest_archive_must_match_member(self):
        r = archive_execution_resolver(self.chain(), {V7: INV[V7], V82: INV[V82], V821: INV[V821]}, INV[V821])
        with self.assertRaises(ValueError): r(manifest(P7, archive=V82))

    def test_non_member_manifest_refused(self):
        r = archive_execution_resolver(self.chain(), {V7: INV[V7], V82: INV[V82], V821: INV[V821]}, INV[V821])
        with self.assertRaises(ValueError): r(manifest(Policy('f' * 64, V7)))

    def test_chain_requires_every_archive_at_startup(self):
        for table in (None, {V82: INV[V82], V821: INV[V821]}):
            with self.assertRaises(ValueError): archive_execution_resolver(self.chain(), table, INV[V821])

    def test_unknown_archive_refused(self):
        t = {V7: INV[V7], V82: INV[V82], V821: INV[V821], 'dd' * 32: INV[V7]}
        with self.assertRaises(ValueError): archive_execution_resolver(self.chain(), t, INV[V821])

    def test_active_entry_must_equal_live_closure(self):
        with self.assertRaises(ValueError):
            archive_execution_resolver(self.chain(), {V7: INV[V7], V82: INV[V82], V821: INV[V82]}, INV[V821])

    def test_malformed_entry_refused(self):
        with self.assertRaises(ValueError):
            archive_execution_resolver(self.chain(), {V7: {'source_files': {}}, V82: INV[V82], V821: INV[V821]}, INV[V821])

    def test_single_policy_uses_live(self):
        r = archive_execution_resolver(P82, None, INV[V82])
        self.assertEqual(r(manifest(P82)), INV[V82])
        with self.assertRaises(ValueError): r(manifest(P82, archive=V7))


if __name__ == '__main__':
    unittest.main()
