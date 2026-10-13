"""Real miner uploads, signatures, native grading and restartable selection.

S3/HTTP and the math session are explicit local fixtures; no model sampling
provenance or H200 qualification is claimed by these coordinator tests.
"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from test_complete_group_fixtures import fixture, FixtureRuntime, digest
from test_complete_group_transport import FakeBucket, FakeHTTP
from subnet.storage import Identity, canonical
from subnet.miner import Miner
from subnet.mimo_coordinator import Coordinator
from subnet.complete_group_transport import CompleteGroupTransport
from subnet.complete_group_native import resolve_task, grade_batch


class NativeTokenizer:
    eos_token_id=15
    def __len__(self):return 16
    def apply_chat_template(self,messages,**kwargs):return [1,int(messages[0]['content'])+2]
    def decode(self,output,**kwargs):
        return {3:r'\boxed{1}',4:r'\boxed{0}',5:'unfinished reasoning'}[output[0]]


class NativeSession:
    def reset(self,index,seed):
        return dict(messages=[dict(role='user',content=str(index))],
            task_hash=digest(dict(index=index,taskset='taskset-1')))
    def step(self,action):
        category='positive'if action['text']==r'\boxed{1}'else'negative'
        return dict(done=True,reward=float(category=='positive'),classification=category,observations=[])
    def close(self):pass


class CoordinatorIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.bucket=FakeBucket();self.http=FakeHTTP(self.bucket)
        self.authority=Identity(bytes([42])*32)
        self.miners=[Identity(bytes([i])*32)for i in (1,2,3)]
        self.tokenizer=NativeTokenizer()
        self.actors=[]
        self.addCleanup(lambda:[a._group_journal.close()for a in self.actors])
        session=patch('subnet.complete_group_native.create_session',side_effect=lambda spec:NativeSession())
        session.start();self.addCleanup(session.stop)
        spec=patch('subnet.complete_group_native.EnvironmentSpec.from_dict',
                   side_effect=lambda value:SimpleNamespace(**value))
        spec.start();self.addCleanup(spec.stop)
        self.manifest=fixture()
        self.manifest['epoch']='private-coordinator-fixture'
        self.manifest['run_id']='private-coordinator-run'
        self.manifest['payable']=False
        self.manifest['max_batches']=9
        self.coordinator=self.make_coordinator()
        self.opening=self.coordinator.open(self.manifest,[m.id for m in self.miners])
        self.opened=self.opening['payload']

    def make_coordinator(self):
        return Coordinator(self.root/'coordinator',self.bucket,self.authority,
            resolve_task=lambda m,e,i:resolve_task(m,e,i,self.tokenizer),
            grade_batch=lambda b,m,miner:grade_batch(b,m,self.tokenizer,miner=miner))

    def actor(self,identity):
        with patch('subnet.model.check_runtime_profile'):
            result=Miner(identity,self.opened,'no-model-loaded',
                state_path=self.root/(identity.id+'.json'),
                complete_group_transport=CompleteGroupTransport(http=self.http))
        self.actors.append(result)
        return result

    def upload(self,actor,index,*,lie=False):
        runtime=FixtureRuntime(self.opened,actor.identity.id)
        def mutate(runtime,row):
            turn=row['turns'][0]
            if lie and row['seed']==0:
                turn['output'][0]=4
                runtime.complete_group_sample_evidence['output_sha256']=digest(turn['output'])
            turn['text']=self.tokenizer.decode(turn['output'])
        runtime.mutate=mutate
        with patch('subnet.miner.make_runtime',return_value=runtime):
            actor.search(index,env_id='math')
        self.assertEqual(actor.upload(),200)
        return self.coordinator.collect(actor.identity.id,now=self.bucket.clock+1)

    def seal(self,**kwargs):
        return self.coordinator.seal(now=self.opened['deadline'],**kwargs)['payload']

    def test_real_upload_native_grading_training_selection_does_not_wait_for_audits(self):
        actor=self.actor(self.miners[0])
        for index in (0,1,2):self.upload(actor,index)
        with patch('subnet.complete_group_verification.verify_rollout',side_effect=AssertionError('no audit barrier')):
            dispatch=self.seal()
        self.assertEqual(dispatch['selected_task_count'],3)
        self.assertEqual(dispatch['selected_rollout_count'],24)
        self.assertFalse(dispatch['training_requires_sampling_audit'])
        self.assertEqual(dispatch['excluded'],[])
        for row in dispatch['selected']:
            grades=row['native_grades']
            self.assertEqual([g['nonce']for g in grades['results']],list(range(8)))
            self.assertFalse(grades['sampling_verified'])
            self.assertEqual([g['classification']for g in grades['results']],['positive','negative']+['neutral']*6)
        self.assertFalse(any('/staging/'in key for key in self.bucket.reads))

    def test_cross_miner_collisions_zero_both_and_keep_unique_task(self):
        first=self.actor(self.miners[0]);second=self.actor(self.miners[1])
        self.upload(first,0);self.upload(second,0);self.upload(second,1)
        dispatch=self.seal()
        self.assertEqual([r['index']for r in dispatch['selected']],[1])
        self.assertEqual(len(dispatch['excluded']),2)
        self.assertEqual({r['reason']for r in dispatch['excluded']},{'cross-miner-task-collision'})

    def test_native_false_claim_excludes_only_bad_batch(self):
        first=self.actor(self.miners[0]);second=self.actor(self.miners[1])
        self.upload(first,0,lie=True);self.upload(second,1)
        dispatch=self.seal()
        self.assertEqual([r['index']for r in dispatch['selected']],[1])
        self.assertEqual(dispatch['excluded'][0]['reason'],'native-claim-invalid')
        self.assertIsNone(dispatch['excluded'][0]['native_grades'])
        self.assertIn('submitted outcome',dispatch['excluded'][0]['detail'])

    def test_infrastructure_failure_freezes_population_and_retries_without_penalty(self):
        actor=self.actor(self.miners[0]);self.upload(actor,0)
        original=self.coordinator.grade_batch
        self.coordinator.grade_batch=lambda *a,**k:(_ for _ in ()).throw(OSError('grader unavailable'))
        with self.assertRaisesRegex(OSError,'grader unavailable'):self.seal()
        self.assertTrue((self.coordinator.root/'POPULATION.json').exists())
        self.assertFalse((self.coordinator.root/'DISPATCH.json').exists())
        with self.assertRaisesRegex(ValueError,'sealed'):
            self.coordinator.collect(actor.identity.id,now=self.opened['deadline'])
        self.coordinator.grade_batch=original
        dispatch=self.seal()
        self.assertEqual(dispatch['selected_task_count'],1)
        self.assertEqual(dispatch['excluded'],[])

    def test_lost_dispatch_ack_recovers_exact_original_bytes_without_regrading(self):
        actor=self.actor(self.miners[0]);self.upload(actor,0)
        original=self.bucket.put_object
        def lose_ack(**kwargs):
            result=original(**kwargs)
            if kwargs['Key'].endswith('/mimo/dispatch.json'):raise OSError('ack lost')
            return result
        with patch.object(self.bucket,'put_object',side_effect=lose_ack):
            with self.assertRaisesRegex(OSError,'ack lost'):self.seal()
        prepared=json.loads((self.coordinator.root/'PREPARED-DISPATCH.json').read_bytes())
        self.assertFalse((self.coordinator.root/'DISPATCH.json').exists())
        restored=self.make_coordinator()
        restored.grade_batch=lambda *a,**k:(_ for _ in ()).throw(AssertionError('must not regrade'))
        result=restored.seal(now=self.opened['deadline'])
        self.assertEqual(result,prepared)
        self.assertEqual(self.bucket.objects[f'private/{self.opened["epoch"]}/mimo/dispatch.json'][0],canonical(result))
        self.assertEqual(restored.seal(now=self.opened['deadline']),result)

    def test_population_recovery_ignores_later_capture_file_replacement(self):
        actor=self.actor(self.miners[0]);self.upload(actor,0)
        with patch.object(self.coordinator,'grade_batch',side_effect=OSError('temporary')):
            with self.assertRaises(OSError):self.seal()
        (self.coordinator.root/'captured'/(actor.identity.id+'.json')).write_text('not valid JSON')
        dispatch=self.seal()
        self.assertEqual(dispatch['selected_task_count'],1)

    def test_native_callback_requires_complete_original_nonce_receipts(self):
        actor=self.actor(self.miners[0]);self.upload(actor,0)
        original=self.coordinator.grade_batch
        def malformed(*a,**k):
            result=original(*a,**k);result['results'][1]['nonce']=0;return result
        with patch.object(self.coordinator,'grade_batch',side_effect=malformed):
            with self.assertRaisesRegex(ValueError,'every original attempt'):self.seal()
        self.assertEqual(self.seal()['selected_task_count'],1)

    def test_task_budget_never_splits_groups_and_policy_cannot_change_on_retry(self):
        actor=self.actor(self.miners[0])
        for index in (0,1,2):self.upload(actor,index)
        dispatch=self.seal(max_tasks=2)
        self.assertEqual(dispatch['selected_task_count'],2)
        self.assertEqual(dispatch['selected_rollout_count'],16)
        self.assertEqual(len(dispatch['excluded']),1)
        self.assertEqual(dispatch['excluded'][0]['reason'],'training-task-budget')
        self.assertEqual(self.seal(max_tasks=2),dispatch)
        with self.assertRaisesRegex(ValueError,'original finalized dispatch'):self.seal(max_tasks=1)

    def test_empty_epoch_can_dispatch_without_minting_training_data(self):
        dispatch=self.seal()
        self.assertEqual(dispatch['selected_task_count'],0)
        self.assertEqual(dispatch['selected'],[])
        self.assertEqual(dispatch['excluded'],[])

    def test_opening_retry_preserves_capabilities_and_rejects_request_change(self):
        restored=self.make_coordinator()
        self.assertEqual(restored.open(self.manifest,[m.id for m in self.miners]),self.opening)
        changed=deepcopy(self.manifest);changed['deadline']+=1
        with self.assertRaisesRegex(ValueError,'cannot change'):
            restored.open(changed,[m.id for m in self.miners])
        with self.assertRaisesRegex(ValueError,'deadline'):
            restored.seal(now=self.opened['deadline']-1)


if __name__=='__main__':unittest.main()
