import copy,hashlib,json,unittest
from unittest.mock import patch
from nacl.signing import SigningKey
from test_committed_training_inputs import LearnerCollectionTests,sign
from subnet import committed_training_inputs as learner
from subnet.storage import canonical

class BoundedTrainingSelection(LearnerCollectionTests):
    def large_controller(self,n=300):
        controller=self.controller();self.manifest['environments'][0]['indices']=list(range(n))
        receipts={}
        for base in range(0,n,3):
            key=SigningKey(hashlib.sha256(str(base).encode()).digest());identity=key.verify_key.encode().hex();children=[];rows=[]
            for slot,index in enumerate(range(base,min(base+3,n))):
                batch=copy.deepcopy(self.batch);batch['index']=batch['sample_index']=index
                for r in batch['rollouts']:r['index']=r['sample_index']=index
                data=canonical(dict(version=learner.ARTIFACT_VERSION,epoch=self.manifest['epoch'],checkpoint=self.manifest['checkpoint']['id'],miner=identity,slot=slot,batch=batch))
                child=dict(slot=slot,env_id='math',index=index,batch_sha256=learner.sha(batch),sha256=hashlib.sha256(b'proof'+str(index).encode()).hexdigest(),size=999,training_sha256=hashlib.sha256(data).hexdigest(),training_size=len(data));children.append(child)
                name='private/frozen/'+str(index);controller.bucket.objects[name]=data;rows.append(dict(slot=slot,sha256=child['training_sha256'],size=len(data),frozen_key=name,captured_at=21))
            payload=dict(version='small-commitment-pairs-v2',epoch=self.manifest['epoch'],miner=identity,checkpoint=self.manifest['checkpoint']['id'],source=self.manifest['source_bundle']['sha256'],batches=children)
            receipts[identity]=dict(commitment_document=sign(key,payload),training_documents=rows)
        controller.gateway.capture_learner=lambda epoch:receipts
        return controller
    def test_300_eligible_remain_auditable_while_training_is_bounded_and_retry_draw_is_stable(self):
        controller=self.large_controller();captured=[]
        def population(document,receipts,round_number,at,authority,*,eligible_pairs):
            captured.append(len(eligible_pairs));return dict(manifest_document=document,receipts=receipts,round=round_number,eligible_evidence_ids=eligible_pairs)
        with patch('subnet.continuous_audit_service.register_population',side_effect=population):
            training,submissions,pop=learner.collect(controller,self.manifest,round_number=16)
            self.assertEqual((len(submissions),pop['training_count'],pop['eligible_count']), (256,256,300))
            self.assertEqual(len(pop['eligible_inventory']),300);self.assertEqual(captured,[300]);self.assertEqual(pop['exclusions'],[])
            self.assertEqual(pop['training_selection']['unselected_count'],44)
            self.assertEqual(training['training_coverage']['seed'],pop['training_selection']['seed'])
            # Simulate a failure before population publication/cache completion;
            # persisted seed/eligible inventory must preserve the exact original draw.
            (self.root/(self.manifest['epoch']+'-learner-population.json')).unlink()
            with patch('secrets.token_hex',side_effect=AssertionError('redraw')):
                repeated=learner.collect(controller,self.manifest,round_number=16)
            self.assertEqual(learner.receipt_inventory(submissions),learner.receipt_inventory(repeated[1]))
            self.assertEqual(training['training_coverage'],repeated[0]['training_coverage'])
    def test_immutable_seed_cannot_be_rebound_to_changed_inventory(self):
        controller=self.large_controller();learner.collect(controller,self.manifest)
        (self.root/(self.manifest['epoch']+'-learner-population.json')).unlink()
        controller.gateway.capture_learner=lambda epoch:{}
        with self.assertRaisesRegex(ValueError,'selection context'):learner.collect(controller,self.manifest)
    def test_bucket_infrastructure_failure_is_not_a_fraud_or_selection_result(self):
        controller=self.large_controller(3);controller.bucket.get=lambda key:(_ for _ in()).throw(ConnectionError('infra'))
        with self.assertRaises(ConnectionError):learner.collect(controller,self.manifest)
        self.assertFalse((self.root/(self.manifest['epoch']+'-learner-population.json')).exists())
        self.assertFalse((self.root/(self.manifest['epoch']+'-learner-training-selection.json')).exists())
