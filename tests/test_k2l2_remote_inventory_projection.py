import copy
import unittest
from ops import durable_learner_service as m

class ProjectionControls(unittest.TestCase):
    def setUp(self):
        self.files={'subnet/module_'+str(i)+'.py':'a'*64 for i in range(179)}
        self.extra={'subnet/source_sampling_admission.py':'b'*64}
        self.versions={'torch':'qualified','transformers':'qualified','toploc':'qualified'}
        self.observed=dict(source_files=dict(self.files,**self.extra),runtime_versions=self.versions)
        self.config={'K':2,'L':2,'max_batches':3,'sampling_policy':{'version':'forced-inverse-cdf-prefill-miner-bound-v5','max_attempts':1000}}
    def test_eight_sample_signed_opening(self):
        config=dict(self.config,K=4,L=4);opening={'K':4,'L':4}
        result=m.calibration_opening_with_quota(lambda a,b,c,d:d,None,config,None,opening)
        self.assertEqual(result,opening)
        with self.assertRaises(ValueError):m.calibration_opening_with_quota(lambda *a:None,None,config,None,dict(opening,K=2))
    def test_projection_authenticates_all_180_before_returning_179(self):
        before=copy.deepcopy(self.observed)
        result=m.normalized_remote_metadata(self.observed,self.files,self.extra,self.versions)
        self.assertEqual(result['source_files'],self.files)
        self.assertEqual(self.observed,before)
        result['source_files'].clear()
        self.assertEqual(len(self.files),179)
    def test_rejects_unqualified_inventory_changes(self):
        for change in ('missing-gate','mutated-gate','extra-file','missing-runtime','changed-runtime','versions'):
            with self.subTest(change=change):
                metadata=copy.deepcopy(self.observed)
                if change=='missing-gate':metadata['source_files'].pop(next(iter(self.extra)))
                elif change=='mutated-gate':metadata['source_files'][next(iter(self.extra))]='c'*64
                elif change=='extra-file':metadata['source_files']['subnet/rogue.py']='c'*64
                elif change=='missing-runtime':metadata['source_files'].pop(next(iter(self.files)))
                elif change=='changed-runtime':metadata['source_files'][next(iter(self.files))]='c'*64
                else:metadata['runtime_versions']['torch']='unqualified'
                with self.assertRaises(ValueError):m.normalized_remote_metadata(metadata,self.files,self.extra,self.versions)
    def test_projection_uses_signed_quota_and_preserves_open_argument(self):
        opening={'K':2,'L':2};before=copy.deepcopy(opening)
        def calibration(controller,config,status,projected):
            self.assertEqual(projected['max_batches'],3)
            return dict(projected,sampling_policy='confirmed')
        result=m.calibration_opening_with_quota(calibration,None,self.config,None,opening)
        self.assertNotIn('max_batches',result)
        self.assertEqual(opening,before)
        self.assertEqual(result['sampling_policy'],'confirmed')
    def test_rejects_quota_conflicts(self):
        for config,opening in ((dict(self.config,max_batches=4),{}),(self.config,{'max_batches':4})):
            with self.assertRaises(ValueError):m.calibration_opening_with_quota(lambda *a:None,None,config,None,opening)
    def test_existing_quota_is_not_removed(self):
        opening={'max_batches':3}
        self.assertEqual(m.calibration_opening_with_quota(lambda a,b,c,d:d,None,self.config,None,opening),opening)
    def test_legacy_contract_is_unchanged(self):
        opening={};config=dict(self.config,sampling_policy={'version':'forced-inverse-cdf-prefill-support-v3'})
        self.assertIs(m.calibration_opening_with_quota(lambda a,b,c,d:d,None,config,None,opening),opening)
if __name__=='__main__':unittest.main()
