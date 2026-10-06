"""Real signed post-commit intent, delayed scheduling and process-loss controls."""
import copy,json,subprocess,sys,threading,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
from test_optimizer_state_cache import StateCacheControls
from subnet.storage import canonical
from subnet.optimizer_state_cache import StateCache,promote
from subnet.remote_backend import RemoteJobs,RemoteObservationTimeout
from subnet.persistent_training_controller import retire_completed_cache

class OriginalACKIntent(unittest.TestCase):
    def setUp(self):
        f=StateCacheControls();f.setUp();self.addCleanup(f.doCleanups);self.f=f
        f.manifest['checkpoint']=dict(id='11'*32,files={})
        f.job['manifest']=f.sign(f.manifest);self.descriptor=f.candidate()
        self.report=json.loads((f.out/'report.json').read_bytes());self.pointer=f.ack['payload']['trainer_state']
        self.remote=RemoteJobs.__new__(RemoteJobs);r=self.remote
        r.workspace=str(f.root);r.python=sys.executable;r.code=str(Path(__file__).resolve().parents[1])
        r.controller=SimpleNamespace(authority=SimpleNamespace(id=f.authority),signed=f.sign)
        r.command=lambda text,timeout:subprocess.check_output(['bash','-c',text],text=True,timeout=timeout)
        r.state=f.root/'roles';r.state.mkdir()
        self.guard=f.root/'.optimizer-state-cache/promotion.json'
    def prepare(self):return self.remote.prepare_training_cache_ack(self.f.job,self.report,self.pointer)
    def test_real_intent_precedes_delayed_cleanup_and_only_genuine_promotion_permits_warm_parent(self):
        f=self.f;started=[];action=Mock()
        controller=SimpleNamespace(state=f.root,authority=SimpleNamespace(id=f.authority),
            jobs=SimpleNamespace(prepare_training_cache_ack=self.remote.prepare_training_cache_ack,retire_training_cache=action))
        fake=SimpleNamespace(is_alive=lambda:False,start=lambda:started.append(json.loads(self.guard.read_bytes())))
        with patch.object(threading,'Thread',return_value=fake):
            retire_completed_cache(controller,f.job,self.report,self.pointer)
        self.assertEqual(started[0]['phase'],'pending');action.assert_not_called()
        with self.assertRaises(RemoteObservationTimeout):self.remote.wait_training_cache_ack(budget=0)
        new=dict(f.job,job_id='next-original')
        with StateCache(f.root,new,f.manifest,f.authority)as cache:
            cache.promotion_wait_seconds=0
            with self.assertRaisesRegex(ValueError,'terminal observation'):cache.prepare_parent(self.descriptor,'aa'*32)
        # Real detached helper uses this exact signed original ACK and original
        # candidate. No fabricated marker or direct ready override is allowed.
        ack=started[0]['ack'];script='DATA='+repr(dict(ack=ack,authority=f.authority,root=str(f.root),code=self.remote.code))+'\n'+'''import sys,json
sys.path.insert(0,DATA['code'])
from subnet.optimizer_state_cache import promote
print(json.dumps(promote(DATA['ack'],DATA['authority'],DATA['root'])))
'''
        receipt=self.remote._durable_cache_ack(script,ack,f.job['job_id'])
        self.assertTrue(receipt['promoted']);self.assertTrue(self.remote.wait_training_cache_ack(budget=0)['ready'])
        with StateCache(f.root,new,f.manifest,f.authority)as cache:self.assertGreater(cache.prepare_parent(self.descriptor,'aa'*32),0)
    def test_synchronous_intent_failure_never_schedules_cleanup_or_returns_success(self):
        controller=SimpleNamespace(state=self.f.root,authority=SimpleNamespace(id=self.f.authority),
            jobs=SimpleNamespace(prepare_training_cache_ack=Mock(side_effect=TimeoutError('lost intent reply')),retire_training_cache=Mock()))
        with patch.object(threading,'Thread')as thread:
            with self.assertRaises(TimeoutError):retire_completed_cache(controller,self.f.job,self.report,self.pointer)
        thread.assert_not_called();self.assertTrue((self.guard.parent/'pending.json').exists())
    def test_missing_prepare_hook_fails_closed_for_cache_enabled_original_completion(self):
        controller=SimpleNamespace(state=self.f.root,authority=SimpleNamespace(id=self.f.authority),jobs=SimpleNamespace(retire_training_cache=Mock()))
        with self.assertRaisesRegex(ValueError,'synchronous original ACK'):retire_completed_cache(controller,self.f.job,self.report,self.pointer)
    def test_lost_prepare_reply_recovers_exact_intent_without_promotion_or_second_handle(self):
        command=self.remote.command
        def lost(text,timeout):command(text,timeout);raise subprocess.TimeoutExpired('SSH',timeout)
        self.remote.command=lost
        with self.assertRaises(subprocess.TimeoutExpired):self.prepare()
        before=self.guard.read_bytes();self.remote.command=command
        self.assertTrue(self.prepare()['idempotent']);self.assertEqual(before,self.guard.read_bytes())
        self.assertFalse(list(self.f.root.glob('*-cache-ACK-*')))
        self.assertFalse((self.guard.parent/'current.json').exists())
    def test_crash_before_intent_rename_recovers_only_identical_signed_ACK(self):
        ack=self.remote.training_cache_ack(self.f.job,self.report,self.pointer)
        tmp=self.guard.parent/'promotion.intent.tmp';tmp.write_bytes(canonical(dict(phase='pending',ack=ack,synchronously_prepared=True)));tmp.chmod(0o600)
        self.assertTrue(self.prepare()['prepared']);self.assertFalse(tmp.exists())
        self.assertEqual(json.loads(self.guard.read_bytes())['ack'],ack)
    def test_conflicting_pending_original_ACK_is_preserved(self):
        self.prepare();before=self.guard.read_bytes();report=copy.deepcopy(self.report);report['unexpected_change']=True
        with self.assertRaises(subprocess.CalledProcessError):self.remote.prepare_training_cache_ack(self.f.job,report,self.pointer)
        self.assertEqual(before,self.guard.read_bytes())
    def test_absent_guard_with_original_pending_candidate_cannot_dispatch_next_job(self):
        with self.assertRaises(RemoteObservationTimeout):self.remote.wait_training_cache_ack(budget=0)
        actual=self.remote.wait_training_cache_ack;self.remote.wait_training_cache_ack=lambda:actual(budget=0)
        with self.assertRaises(RemoteObservationTimeout):self.remote.run('next-original','train',self.f.manifest)
        self.assertEqual(list(self.remote.state.iterdir()),[])
    def test_worker_does_not_discard_actual_approved_parent_candidate_before_ACK_intent(self):
        with StateCache(self.f.root,dict(self.f.job,job_id='next-original'),self.f.manifest,self.f.authority)as cache:
            with self.assertRaisesRegex(ValueError,'awaits original durability ACK'):cache.prepare_parent(self.descriptor,'aa'*32)
        self.assertTrue(list((self.guard.parent/'candidate-original').glob('*.safetensors')))
        self.assertTrue((self.guard.parent/'pending.json').exists())

if __name__=='__main__':unittest.main()
