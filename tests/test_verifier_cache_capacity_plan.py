import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from ops.verifier_cache_capacity_plan import canonical_digest, plan_reuse


class CapacityPlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.a, self.b = self.root/'keeper', self.root/'runtime'
        self.a.mkdir(); self.b.mkdir()
        self.files = {}
        for name, data in [('config.json', b'{}'), ('model.safetensors', b'weights'*2048)]:
            self.files[name] = dict(size=len(data), sha256=hashlib.sha256(data).hexdigest())
            (self.a/name).write_bytes(data); (self.b/name).write_bytes(data)
        self.cp = canonical_digest({n:m['sha256'] for n,m in self.files.items()})
        self.pending = 'a'*64

    def plan(self, **kw):
        return plan_reuse(self.cp, self.files, self.a, self.b, [self.cp,self.pending],
                         free_bytes=10, artifact_bytes=100, reserve_bytes=100, **kw)

    def test_real_duplicate_inode_capacity_proof_is_read_only(self):
        before = {p: (p.stat().st_ino,p.read_bytes()) for root in (self.a,self.b) for p in root.iterdir()}
        result = self.plan()
        self.assertGreater(result['potential_reclaimed_bytes'], 0)
        self.assertFalse(result['mutation_authorized'])
        self.assertFalse(result['runtime_idle_verified'])
        self.assertEqual(result['protected_checkpoints'], [self.cp,self.pending])
        self.assertEqual(before, {p: (p.stat().st_ino,p.read_bytes()) for p in before})

    def test_existing_hardlinks_have_zero_reclaim(self):
        for name in self.files:
            (self.b/name).unlink(); os.link(self.a/name,self.b/name)
        self.assertEqual(self.plan()['potential_reclaimed_bytes'], 0)

    def test_unknown_destination_alias_never_claims_reclaimed_capacity(self):
        os.link(self.b/'model.safetensors',self.root/'unknown-alias')
        result = self.plan()
        self.assertFalse(result['single_link_replacement_candidate'])
        self.assertEqual(result['potential_reclaimed_bytes'],0)
        self.assertFalse(result['sufficient_capacity_after_prospective_reuse'])

    def test_changed_bytes_refuse(self):
        p=self.b/'model.safetensors'; p.write_bytes(b'X'*self.files[p.name]['size'])
        with self.assertRaisesRegex(ValueError,'byte hash'): self.plan()

    def test_symlink_and_extra_file_refuse(self):
        p=self.b/'model.safetensors'; p.unlink(); p.symlink_to(self.a/p.name)
        with self.assertRaisesRegex(ValueError,'ordinary'): self.plan()
        p.unlink(); p.write_bytes((self.a/p.name).read_bytes()); (self.b/'extra').write_text('x')
        with self.assertRaisesRegex(ValueError,'membership'): self.plan()

    def test_bool_budget_and_missing_protection_refuse(self):
        with self.assertRaises(ValueError):
            plan_reuse(self.cp,self.files,self.a,self.b,[self.cp],free_bytes=True,artifact_bytes=0,reserve_bytes=0)
        with self.assertRaises(ValueError):
            plan_reuse(self.cp,self.files,self.a,self.b,[self.pending],free_bytes=0,artifact_bytes=0,reserve_bytes=0)

if __name__ == '__main__': unittest.main()
