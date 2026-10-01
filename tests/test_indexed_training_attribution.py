import copy
import hashlib
import unittest
from ops.check_gpu_continuous_evidence import training_harness_digest
from subnet.sample_harness import VERSION,project,resolve
from subnet.storage import canonical

class IndexedTrainingAttribution(unittest.TestCase):
    def setUp(self):
        def row(text):return dict(version='text-tools-v1',policy='candidates',
            candidates=[text,'wrong'],max_output_tokens=16,temperature=1.,top_p=1.)
        self.wrapper=dict(version=VERSION,by_index={'0':row('first'),'1':row('second')})
        self.manifest=dict(environments=[dict(env_id='original',spec={},indices=[],
            harness=project(self.wrapper,[],[0,1]))],
            sample_harness_registry={'original':dict(indices=[0,1],harness=copy.deepcopy(self.wrapper))},
            heldout_indices={'original':[2]})
        self.pos=dict(env_id='original',index=0,harness={'candidates':['untrusted']})
        self.neg=dict(env_id='original',index=0)

    def test_inactive_family_resolves_full_signed_registry(self):
        expected=hashlib.sha256(canonical(resolve(self.wrapper,0,[0,1]))).hexdigest()
        self.assertEqual(training_harness_digest(self.manifest,self.pos,self.neg),expected)
        self.pos['index']=self.neg['index']=1
        self.assertNotEqual(training_harness_digest(self.manifest,self.pos,self.neg),expected)

    def test_live_projection_cannot_override_full_signed_approval(self):
        row=self.manifest['environments'][0]
        row.update(indices=[0],harness=project(self.wrapper,[0],[0,1]))
        row['harness']['by_index']['0']['candidates'][0]='forged'
        with self.assertRaises(ValueError):training_harness_digest(self.manifest,self.pos,self.neg)

    def test_unapproved_heldout_or_mismatched_index_refused(self):
        for index in [2,-1,True,'0']:
            with self.subTest(index=index),self.assertRaises(ValueError):
                training_harness_digest(self.manifest,dict(env_id='original',index=index),dict(env_id='original',index=index))
        with self.assertRaises(ValueError):training_harness_digest(self.manifest,self.pos,dict(env_id='other',index=0))
        self.manifest['heldout_indices']['original'].append(0)
        with self.assertRaises(ValueError):training_harness_digest(self.manifest,self.pos,self.neg)

    def test_missing_full_registry_refused_and_plain_history_unchanged(self):
        m=copy.deepcopy(self.manifest);m.pop('sample_harness_registry')
        with self.assertRaises(ValueError):training_harness_digest(m,self.pos,self.neg)
        m['environments'][0]['harness']=self.wrapper['by_index']['0']
        self.assertIsNone(training_harness_digest(m,self.pos,self.neg))

if __name__=='__main__':unittest.main()
