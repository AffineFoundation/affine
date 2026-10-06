import copy
import json
import unittest
from pathlib import Path

from ops import continuous_owned_heldout128 as service
import test_owned_cached_group_operator as fixture


class Adapter:
    def __init__(self, test):
        self.t=test; self.launches=[]; self.lost_reply=False; self.reserved=False
        self.result={'status':'observing-original'}; self.tamper=False;self.pointer_calls=0
    def idle(self): return not self.reserved
    def bootstrap_complete(self, entry):return copy.deepcopy(self.t.bootstrap)
    def publication(self, row):
        t=self.t;d=dict(inference_checkpoint=t.f.cp['id'],optimizer_steps=14,epoch='epoch-23')
        return dict(checkpoint_descriptor=t.f.sign(t.f.cp),optimizer_publication=t.f.sign(dict(
            version='authority-persistent-trainer-state-v1',descriptor=d,descriptor_sha256=service.digest(d))))
    def prepare(self, policy, publication, identity):
        t=self.t;scope=copy.deepcopy(t.f.scope);scope['workspace']=str(t.f.root)
        scope.update(source_path='/qualified/source',endpoint=dict(workspace=scope['workspace'],code='/qualified/source',python='/approved/python',known_hosts='/approved/knownhosts'))
        if self.tamper:scope['runtime_versions']={'fixture':'wrong'}
        if getattr(self,'stale_endpoint',False):scope['endpoint']['workspace']='/old/CP13'
        return dict(checkpoint=t.f.cp['id'],cohort_sha256=policy['cohort_sha256'],identity=identity,
                    original_jobs=t.f.jobs,scope=t.f.sign(scope),workspace=scope['workspace'],expires_at=1000,optimizer_step=14)
    def launch(self, packet):
        self.launches.append(copy.deepcopy(packet))
        if self.lost_reply:raise TimeoutError('lost detached launch reply')
    def observe(self, packet):return copy.deepcopy(self.result)
    def publish_pointer(self,packet,summary):self.pointer_calls+=1
    def archive_complete(self,packet,result):
        t=self.t
        return t.f.sign(dict(version='owned-cached-heldout128-checkpoint-actual-v1',checkpoint=t.f.cp,
            cohort_sha256=t.policy['cohort_sha256'],source_sha256=service.SOURCE,
            all_four_genuine_full_R2_ACKs=True,group_owned_model_retired=True,task_count=128,
            successes=result['successes']))


class ContinuousTests(unittest.TestCase):
    def setUp(self):
        self.f=fixture.OperatorTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.production=self.f.root/'production';self.production.mkdir();self.journal=self.f.root/'service'
        self.now=10
        self.policy=dict(version=service.VERSION,execute_allowed=True,created_at=1,expires_at=10000,
            first_optimizer_step=14,source_sha256=service.SOURCE,source_files=self.f.files,
            runtime_versions=self.f.scope['runtime_versions'],per_checkpoint_job_count=4,per_original_task_count=32,
            group_lifetime_seconds=1000,groups=self.f.groups,cohort_sha256=service.digest(self.f.groups),
            mining_indices=list(range(6746)),old32_indices=list(range(7000,7032)),
            production_directory=str(self.production),journal_directory=str(self.journal),
            adapter_path=str(Path(__file__).resolve()),production_services_stopped=False,normal32_replaced=False)
        self.bootstrap=self.f.sign(dict(version='owned-cached-heldout128-checkpoint-actual-v1',checkpoint='b'*64,
            source_sha256=service.SOURCE,cohort_sha256=self.policy['cohort_sha256'],
            all_four_genuine_full_R2_ACKs=True,group_owned_model_retired=True,task_count=128))
        self.policy['bootstrap_completed']=[dict(checkpoint='b'*64,optimizer_step=13,summary_sha256=service.digest(self.bootstrap))]
        self.completion=dict(epoch='epoch-23',round=23,completed_at=5,next_checkpoint=self.f.cp['id'])
        self.write_completion();self.adapter=Adapter(self)
    def write_completion(self):
        p=self.production/'epoch-23-signed-learner-completion.json';p.write_text(json.dumps(self.f.sign(self.completion)))
    def loop(self):return service.Continuous128(self.f.sign(self.policy),self.f.authority,self.adapter,clock=lambda:self.now)
    def test_published_checkpoint_four_originals_restart_complete_once(self):
        with self.loop() as loop:
            self.assertEqual(loop.step()['status'],'prepared-originals')
            self.assertEqual(loop.step()['status'],'observing-original')
        self.adapter.result=dict(status='complete',durable_ACK_count=4,owned_model_retired=True,count=128,successes=84)
        with self.loop() as loop:
            self.assertEqual(loop.step()['successes'],84)
            self.assertEqual(loop.step()['status'],'complete128-projection-published')
            self.assertEqual(loop.step()['status'],'waiting-published-checkpoint')
        self.assertEqual(self.adapter.pointer_calls,1)
        self.assertEqual(len(self.adapter.launches),1)
    def test_lost_reply_never_relaunches_original(self):
        with self.loop() as loop:
            loop.step();self.adapter.lost_reply=True
            with self.assertRaises(TimeoutError):loop.step()
        self.adapter.lost_reply=False
        with self.loop() as loop:
            self.assertEqual(loop.step()['status'],'observing-original')
        self.assertEqual(len(self.adapter.launches),1)
    def test_partial_ACK_no_retirement_or_wrong_count_never_score(self):
        with self.loop() as loop:
            loop.step();loop.step()
            for change in ({'durable_ACK_count':3},{'owned_model_retired':False},{'count':96}):
                self.adapter.result=dict(status='complete',durable_ACK_count=4,owned_model_retired=True,count=128,successes=84)|change
                with self.assertRaises(ValueError):loop.step()
            self.assertNotIn('summary',loop.state['checkpoints'][self.f.cp['id']])
    def test_infrastructure_failure_retains_original_no_zero_no_new_group(self):
        with self.loop() as loop:
            loop.step();loop.step();self.adapter.result=dict(status='original-infrastructure-failure',model_reward=None)
            loop.step();self.assertEqual(loop.step()['status'],'original-infrastructure-reconciliation-required')
            self.assertNotIn('summary',loop.state['checkpoints'][self.f.cp['id']])
        self.assertEqual(len(self.adapter.launches),1)
    def test_future_or_unsigned_or_changed_publication_refused(self):
        self.completion['completed_at']=11;self.write_completion()
        with self.loop() as loop,self.assertRaises(ValueError):loop.step()
        self.completion['completed_at']=5;self.write_completion()
        original=self.adapter.publication
        def wrong(row):
            v=original(row);v['checkpoint_descriptor']['payload']['id']='a'*64;return v
        self.adapter.publication=wrong
        with self.loop() as loop,self.assertRaises(Exception):loop.step()
        self.assertFalse(self.adapter.launches)
    def test_exact_source_runtime_cohort_and_leakage_reject(self):
        for key,value in [('source_sha256','wrong'),('cohort_sha256','wrong'),('old32_indices',list(range(6746,6778))),('execute_allowed',1)]:
            old=self.policy[key];self.policy[key]=value
            with self.assertRaises(ValueError):self.loop()
            self.policy[key]=old
        self.adapter.tamper=True
        with self.loop() as loop,self.assertRaises(ValueError):loop.step()
    def test_stale_CP13_endpoint_workspace_refused_before_GPU_dispatch(self):
        self.adapter.stale_endpoint=True
        with self.loop() as loop,self.assertRaisesRegex(ValueError,'physical group route'):loop.step()
        self.assertFalse(self.adapter.launches)
    def test_physical_reservation_and_expired_unissued_cannot_dispatch(self):
        with self.loop() as loop:
            loop.step();self.adapter.reserved=True
            self.assertEqual(loop.step()['status'],'physical-reservation-deferred')
            self.adapter.reserved=False;self.now=1001
            self.assertEqual(loop.step()['status'],'expired-unissued')
        self.assertFalse(self.adapter.launches)
    def test_completed_bootstrap_adopts_history_and_active_predecessor_refuses(self):
        with self.loop() as loop:
            self.assertTrue(loop.state['checkpoints']['b'*64]['bootstrap'])
        self.policy['bootstrap_completed'][0]['summary_sha256']='0'*64
        # Fresh namespace needed to exercise bootstrap rather than bind rejection.
        self.policy['journal_directory']=str(self.f.root/'fresh-service')
        with self.assertRaises(ValueError):
            with self.loop():pass
    def test_no_predecessor_fullACK_or_inflight_summary_can_bootstrap(self):
        self.bootstrap['payload']['all_four_genuine_full_R2_ACKs']=False
        self.bootstrap=self.f.sign(self.bootstrap['payload'])
        self.policy['bootstrap_completed'][0]['summary_sha256']=service.digest(self.bootstrap)
        with self.assertRaises(ValueError):
            with self.loop():pass
    def test_paired_legacy_bootstrap_preserves_original_envelope_without_new_jobs(self):
        self.bootstrap=self.f.sign(dict(version='owned-cached-heldout128-paired-actual-v1',
            source_sha256=service.SOURCE,cohort_sha256=self.policy['cohort_sha256'],
            all_eight_genuine_full_R2_ACKs=True,all_group_owned_models_retired=True,
            task_count_per_checkpoint=128,groups={'CP13':{}}))
        self.policy['bootstrap_completed'][0].update(group_label='CP13',summary_sha256=service.digest(self.bootstrap))
        with self.loop() as loop:
            self.assertEqual(loop.state['checkpoints']['b'*64]['summary'],self.bootstrap)
        self.assertFalse(self.adapter.launches)
    def test_lost_projection_reply_resumes_pointer_only_not_GPU_group(self):
        with self.loop() as loop:
            loop.step();loop.step();self.adapter.result=dict(status='complete',durable_ACK_count=4,owned_model_retired=True,count=128,successes=84)
            loop.step()
        def lost(*a):raise TimeoutError('lost projection reply')
        self.adapter.publish_pointer=lost
        with self.loop() as loop,self.assertRaises(TimeoutError):loop.step()
        self.adapter.publish_pointer=lambda *a:None
        with self.loop() as loop:self.assertEqual(loop.step()['status'],'complete128-projection-published')
        self.assertEqual(len(self.adapter.launches),1)
    def test_double_service_owner_fails_exclusive_lock(self):
        with self.loop():
            with self.assertRaises(BlockingIOError):
                with self.loop():pass


if __name__=='__main__':unittest.main()
