"""New peer quotas remain manifest-driven; old177/179 grants remain distinct."""
import copy,hashlib,unittest
from pathlib import Path
from test_learner_selection_operator_bridge import PeerTests
from training_receipt_fixtures import sign
class ManifestCPUQuotas(PeerTests):
    def setUp(self):
        super().setUp()
        self.base=Path(__file__).resolve().parents[1]
        additions={'subnet/batch_quotas.py','subnet/sampling_uniqueness.py','subnet/trajectory_identity.py'}
        names=set(self.approval['payload']['scientific_source_files'])
        for n in additions:names.add(n)
        for n in sorted(names-additions):
            if len(names)<=180:break
            names.remove(n)
        for p in sorted((self.base/'subnet').glob('*.py')):
            if len(names)>=180:break
            names.add('subnet/'+p.name)
        self.runtime={n:hashlib.sha256((self.base/n).read_bytes()).hexdigest()for n in names}
        assert len(self.runtime)==180
        self.grant=copy.deepcopy(self.approval['payload']);self.grant.update(version=self.B.K2L2_AUTH_VERSION,scientific_source_files=self.runtime)
        self.manifest=dict(self.m,max_batches=3,sampling_contract={'version':'forced-inverse-cdf-prefill-miner-bound-v5','max_attempts':1000})
    def test_balanced_general_quotas_derive_from_approved_manifest(self):
        for quota in (2,3,4,8,64):
            m=dict(self.manifest,K=quota,L=quota,samples_per_batch=2*quota)
            self.assertEqual(self.B.approval(sign(self.key,self.grant),self.auth,m),self.grant)
    def test_invalid_or_conflicting_quotas_fail(self):
        for change in ({'K':True,'L':True},{'K':4,'L':2},{'K':65,'L':65},{'K':4,'L':4,'samples_per_batch':4},{'K':4,'L':4,'max_batches':257},{'K':4,'L':4,'max_batches':True}):
            with self.subTest(change=change),self.assertRaises(ValueError):self.B.approval(sign(self.key,self.grant),self.auth,dict(self.manifest,**change))
    def test_historical179_does_not_claim_newgeometry(self):
        g=copy.deepcopy(self.grant);g['scientific_source_files'].pop('subnet/batch_quotas.py')
        m=dict(self.manifest,K=2,L=2)
        self.assertEqual(self.B.approval(sign(self.key,g),self.auth,m),g)
        with self.assertRaises(ValueError):self.B.approval(sign(self.key,g),self.auth,dict(m,K=4,L=4))
    def test_new_bridge_preserves_frozen_profile_manifest_grants(self):
        # The historical operator profile is immutable. New execution/selection
        # features do not require relabeling or replacing its approved bytes.
        from subnet import learner_selection_operator_bridge as current
        grant=sign(self.key,self.grant)
        for cap in (1,3,9,256):
            manifest=dict(self.manifest,K=4,L=4,max_batches=cap)
            self.assertEqual(current.approval(grant,self.auth,manifest),
                             self.B.approval(grant,self.auth,manifest))
