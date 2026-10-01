import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ops.import_gpu_pilot_evaluations import canonical, records, sha


class GPUEvaluationImportTests(unittest.TestCase):
    def fixture(self, root):
        report=dict(success=True, stage='complete', after_independent_verified=True,
                    pair_audits={'positive':True,'negative':True},
                    approved_checkpoint_files={'model.safetensors':'a'*64},
                    new_checkpoint={'files':{'model.safetensors':'b'*64}},
                    training={'weights_changed':True,'steps':1,'objective':'head preference','full_model_finetune':False},
                    training_indices=[0], heldout_indices=[2],
                    environment={'id':'original-env','version':'v1'},
                    harness={'version':'plain-transcript-v1','temperature':4},
                    model='reviewed/model',profile={'version':'cuda-pinned'})
        for files, key in ((report['approved_checkpoint_files'],'approved_checkpoint_id'),
                           (report['new_checkpoint']['files'],'id')):
            target=report if key=='approved_checkpoint_id' else report['new_checkpoint']
            target[key]=hashlib.sha256(canonical(files)).hexdigest()
        timestamps={}
        for phase,timestamp in (('before',100),('after',200)):
            name='heldout-'+phase+'-2'
            doc={'index':2,'task_hash':'task-2','seed':202,'reward':0.,'classification':'negative',
                 'turns':[{'output':[0,1],'proofs':['cHJvb2Y=']}]}
            (root/(name+'.json')).write_text(json.dumps(doc))
            np.savez_compressed(root/(name+'.npz'),turn_0=np.zeros((2,3),dtype=np.float32))
            report['heldout_'+phase]=[dict(name=name,index=2,task_hash='task-2',reward=0.,classification='negative',verified=True,
                sha256={suffix:sha(root/(name+suffix)) for suffix in ('.json','.npz')})]
            timestamps[name+'.npz']=timestamp
        return report,timestamps

    def test_real_artifact_binding_and_separate_checkpoint_comparison(self):
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw);report,timestamps=self.fixture(root);points=records(report,root,timestamps)
            self.assertEqual(len(points),2)
            self.assertEqual(points[0]['dataset_id'],points[1]['dataset_id'])
            self.assertNotEqual(points[0]['checkpoint'],points[1]['checkpoint'])
            self.assertEqual([p['mean_reward'] for p in points],[0.,0.])
            self.assertEqual([p['training_steps'] for p in points],[0,1])
            self.assertFalse(points[1]['full_model_finetune'])

    def test_changed_copied_artifact_fails(self):
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw);report,timestamps=self.fixture(root)
            (root/'heldout-after-2.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'artifact hash'):records(report,root,timestamps)

    def test_same_task_different_seed_is_not_comparable(self):
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw);report,timestamps=self.fixture(root)
            path=root/'heldout-after-2.json';doc=json.loads(path.read_text());doc['seed']=999;path.write_text(json.dumps(doc))
            report['heldout_after'][0]['sha256']['.json']=sha(path)
            with self.assertRaisesRegex(ValueError,'incomparable'):records(report,root,timestamps)

    def test_heldout_training_overlap_fails(self):
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw);report,timestamps=self.fixture(root);report['training_indices']=[2]
            with self.assertRaisesRegex(ValueError,'overlap'):records(report,root,timestamps)


if __name__=='__main__':unittest.main()
