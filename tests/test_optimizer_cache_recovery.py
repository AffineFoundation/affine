import copy,json,subprocess,sys,unittest
from pathlib import Path
from unittest.mock import patch
from test_optimizer_state_cache import StateCacheControls
from subnet.optimizer_state_cache import StateCache,promote
from subnet.storage import canonical

class CacheACKRecovery(StateCacheControls):
    def test_idempotent_same_confirmed_current_without_pending(self):
        self.promoted()
        with patch('subnet.optimizer_state_cache.hash_file',side_effect=AssertionError('retry must not rehash')):
            receipt=promote(self.ack,self.authority,self.root)
        self.assertTrue(receipt['idempotent'])
    def test_crash_after_current_promotion_before_pending_unlink(self):
        self.candidate();original=Path.unlink
        def crash(path,*args,**kwargs):
            if path.name=='pending.json':raise RuntimeError('simulated crash')
            return original(path,*args,**kwargs)
        with patch.object(Path,'unlink',crash):
            with self.assertRaisesRegex(RuntimeError,'simulated crash'):promote(self.ack,self.authority,self.root)
        self.assertTrue(promote(self.ack,self.authority,self.root)['idempotent'])
        self.assertFalse((self.root/'.optimizer-state-cache/pending.json').exists())
    def test_current_ack_wrong_lineage_rejects_retry(self):
        self.promoted();marker=self.root/'.optimizer-state-cache/current.json';current=json.loads(marker.read_bytes())
        current['ROOT_ack']['payload']['trainer_state']['namespace']='different';current['ROOT_ack']=self.sign(current['ROOT_ack']['payload']);marker.write_bytes(canonical(current))
        with self.assertRaisesRegex(ValueError,'confirmed lineage|authenticated original optimizer cache lineage'):promote(self.ack,self.authority,self.root)
    def test_pending_promotion_guard_prevents_next_original_from_discarding(self):
        descriptor=self.candidate();guard=self.root/'.optimizer-state-cache/promotion.json';guard.write_bytes(canonical(dict(phase='pending',ack=self.ack)))
        newer=copy.deepcopy(self.job);newer['job_id']='next'
        with StateCache(self.root,newer,self.manifest,self.authority)as cache:
            cache.promotion_wait_seconds=0
            with self.assertRaisesRegex(ValueError,'terminal observation'):cache.prepare_parent(descriptor,'aa'*32)
        self.assertTrue((self.root/'.optimizer-state-cache/pending.json').exists())

    def test_late_pending_promotion_releases_lease_then_continues_same_training(self):
        import threading,time
        descriptor=self.candidate();guard=self.root/'.optimizer-state-cache/promotion.json'
        guard.write_bytes(canonical(dict(phase='pending',ack=self.ack)))
        completed=[]
        def original_promotion():
            time.sleep(.05)
            # The original helper must be able to acquire the same owned lease.
            completed.append(promote(self.ack,self.authority,self.root))
            guard.write_bytes(canonical(dict(phase='complete',ack=self.ack)))
        worker=threading.Thread(target=original_promotion)
        newer=copy.deepcopy(self.job);newer['job_id']='next'
        with StateCache(self.root,newer,self.manifest,self.authority)as cache:
            cache.promotion_wait_seconds=3;worker.start()
            self.assertGreater(cache.prepare_parent(descriptor,'aa'*32),0)
        worker.join(3);self.assertEqual(len(completed),1);self.assertTrue(completed[0]['promoted'])

    def test_confirmed_terminal_failed_promotion_falls_back_cold_preserving_ACK_evidence(self):
        descriptor=self.candidate();guard=self.root/'.optimizer-state-cache/promotion.json'
        guard.write_bytes(canonical(dict(phase='failed',ack=self.ack,child_pid=99999999,child_ticks='1',child_terminal_confirmed=True)))
        newer=copy.deepcopy(self.job);newer['job_id']='next'
        with StateCache(self.root,newer,self.manifest,self.authority)as cache:
            self.assertEqual(cache.prepare_parent(descriptor,'aa'*32),0)
            self.assertEqual(cache.cache_evidence[-1]['reason'],'confirmed-terminal-promotion-failure')
        self.assertFalse(guard.exists());self.assertTrue((self.root/'.optimizer-state-cache/failed-promotion-original.json').exists());self.assertTrue((self.root/'original.json').exists());self.assertTrue((self.out/'report.json').exists())
        self.assertTrue((self.root/'.optimizer-state-cache/failed-original-pending.json').exists())
        self.assertFalse(list((self.root/'.optimizer-state-cache/candidate-original').glob('*.safetensors')))
        self.assertTrue(self.objects)
    def test_consumed_terminal_failure_does_not_poison_later_approved_parent(self):
        descriptor=self.candidate();guard=self.root/'.optimizer-state-cache/promotion.json'
        failure=dict(phase='failed',ack=self.ack,child_pid=99999999,child_ticks='1',child_terminal_confirmed=True)
        guard.write_bytes(canonical(failure))
        newer=copy.deepcopy(self.job);newer['job_id']='next'
        with StateCache(self.root,newer,self.manifest,self.authority)as cache:
            self.assertEqual(cache.prepare_parent(descriptor,'aa'*32),0)
        later=copy.deepcopy(descriptor);later['optimizer_steps']+=1
        newest=copy.deepcopy(self.job);newest['job_id']='following'
        with StateCache(self.root,newest,self.manifest,self.authority)as cache:
            self.assertEqual(cache.prepare_parent(later,'aa'*32),0)
            self.assertEqual(cache.cache_evidence[-1]['reason'],'no-promoted-cache')
        archived=json.loads((self.root/'.optimizer-state-cache/failed-promotion-original.json').read_bytes())
        self.assertEqual(archived,failure);self.assertFalse(guard.exists());self.assertTrue(self.objects)

    def test_fast_child_without_captured_ticks_requires_confirmed_exit_and_absent_pid(self):
        descriptor=self.candidate();guard=self.root/'.optimizer-state-cache/promotion.json'
        guard.write_bytes(canonical(dict(phase='failed',ack=self.ack,child_pid=99999999,child_ticks=None,child_exit_code=1,child_terminal_confirmed=True)))
        with StateCache(self.root,self.job,self.manifest,self.authority)as cache:
            self.assertEqual(cache.prepare_parent(descriptor,'aa'*32),0)

    def test_unknown_terminal_failure_cannot_discard_candidate(self):
        descriptor=self.candidate();guard=self.root/'.optimizer-state-cache/promotion.json'
        guard.write_bytes(canonical(dict(phase='failed',ack=self.ack,child_terminal_confirmed=False)))
        with StateCache(self.root,self.job,self.manifest,self.authority)as cache:
            cache.promotion_wait_seconds=0
            with self.assertRaisesRegex(ValueError,'terminal observation'):cache.prepare_parent(descriptor,'aa'*32)
        self.assertTrue(list((self.root/'.optimizer-state-cache/candidate-original').glob('*.safetensors')))
    def test_live_child_even_claimed_terminal_cannot_trigger_cold_retirement(self):
        import os
        descriptor=self.candidate();ticks=Path('/proc',str(os.getpid()),'stat').read_text().rsplit(')',1)[1].split()[19]
        guard=self.root/'.optimizer-state-cache/promotion.json';guard.write_bytes(canonical(dict(phase='failed',ack=self.ack,child_pid=os.getpid(),child_ticks=ticks,child_terminal_confirmed=True)))
        with StateCache(self.root,self.job,self.manifest,self.authority)as cache:
            with self.assertRaisesRegex(ValueError,'still live'):cache.prepare_parent(descriptor,'aa'*32)
        self.assertTrue(list((self.root/'.optimizer-state-cache/candidate-original').glob('*.safetensors')))
    def test_predispatch_pending_timeout_does_not_allocate_original_training_job(self):
        from types import SimpleNamespace
        from subnet.remote_backend import RemoteJobs,RemoteObservationTimeout
        self.candidate();guard=self.root/'.optimizer-state-cache/promotion.json';guard.write_bytes(canonical(dict(phase='pending',ack=self.ack)))
        remote=RemoteJobs.__new__(RemoteJobs);remote.workspace=str(self.root);remote.python=sys.executable;remote.code=str(Path(__file__).resolve().parents[1])
        remote.controller=SimpleNamespace(authority=SimpleNamespace(id=self.authority))
        remote.command=lambda text,timeout:subprocess.check_output(['bash','-c',text],text=True,timeout=timeout)
        with self.assertRaises(RemoteObservationTimeout):remote.wait_training_cache_ack(budget=0)
        remote.state=self.root/'roles';remote.state.mkdir()
        actual_wait=remote.wait_training_cache_ack;remote.wait_training_cache_ack=lambda:actual_wait(budget=0)
        with self.assertRaises(RemoteObservationTimeout):remote.run('next','train',self.manifest)
        self.assertEqual(list(remote.state.iterdir()),[])
        remote.wait_training_cache_ack=actual_wait
        guard.write_bytes(canonical(dict(phase='failed',ack=self.ack,child_pid=99999999,child_ticks='1',child_terminal_confirmed=True)))
        self.assertTrue(remote.wait_training_cache_ack(budget=0)['ready'])
        self.assertEqual([p.name for p in self.root.glob('*.json')],['original.json'])

class FreshCacheLoader(unittest.TestCase):
    def test_real_fresh_process_pure_admission_then_authenticated_execution_modules(self):
        root=Path(__file__).resolve().parents[1]
        from test_persistent_training_integration import PersistentIntegrationTests
        fixture=PersistentIntegrationTests();fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        manifest=copy.deepcopy(fixture.manifest)
        manifest['optimizer_state_local_cache']=dict(version='sole-current-fp32-state-cache-v1',max_checkpoint_bytes=1000)
        manifest['persistent_publication_policy']=dict(version='parallel-persistent-publication-v1',state_readback='qualified-remote-full',checkpoint_readback_workers=1)
        job=fixture.job(manifest)
        from subnet.persistent_training_protocol import CACHE_EXECUTION_FILES
        import hashlib
        for name in (*job['source_files'],*CACHE_EXECUTION_FILES):job['source_files'][name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
        request=dict(job=job,manifest=manifest,authority=fixture.authority)
        script=r'''
import sys,json
from pathlib import Path
from subnet.persistent_training_protocol import optimizer_cache_policy,CACHE_EXECUTION_FILES,EXECUTION_FILES,validate_job
request=json.load(sys.stdin)
validate_job(request['job'],request['manifest'],request['authority'])
assert 'subnet.optimizer_state_cache' not in sys.modules
assert 'subnet.cache_lifecycle' not in sys.modules
optimizer_cache_policy(dict(optimizer_state_local_cache=dict(version='sole-current-fp32-state-cache-v1',max_checkpoint_bytes=1000),persistent_publication_policy=dict(state_readback='qualified-remote-full')))
assert 'subnet.optimizer_state_cache' not in sys.modules
assert 'subnet.cache_lifecycle' not in sys.modules
from subnet.backend_jobs import install_source_loader,FreshSourceFinder,digest
root=Path.cwd()
files=(*EXECUTION_FILES,*CACHE_EXECUTION_FILES)
# Actual source authentication is completed before the loader is installed.
for name in files:assert digest(root/name)==__import__('hashlib').sha256((root/name).read_bytes()).hexdigest()
install_source_loader(root,files)
import subnet.optimizer_state_cache as cache
import subnet.cache_lifecycle as lifecycle
assert isinstance(cache.__spec__.loader, __import__('importlib').abc.Loader)
assert cache.__spec__.loader.__class__.__qualname__.startswith('FreshSourceFinder.')
assert lifecycle.__spec__.loader.__class__.__qualname__.startswith('FreshSourceFinder.')
print('fresh-cache-loader-ok')
'''
        output=subprocess.check_output([sys.executable,'-B','-c',script],cwd=root,text=True,input=json.dumps(request))
        self.assertIn('fresh-cache-loader-ok',output)

class DurableACKTransport(unittest.TestCase):
    def test_lost_launch_response_recovers_original_supervisor_without_second_execution(self):
        import tempfile,time
        from types import SimpleNamespace
        from subnet.remote_backend import RemoteJobs
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);(root/'.optimizer-state-cache').mkdir();counter=root/'count'
            script='from pathlib import Path;import json,time;p=Path('+repr(str(counter))+');p.write_text("one");time.sleep(0.1);print(json.dumps(dict(status="complete",removed_checkpoints=[])))'
            remote=RemoteJobs.__new__(RemoteJobs);remote.workspace=directory;remote.python=sys.executable
            calls=[]
            def command(text,timeout):
                output=subprocess.check_output(['bash','-c',text],text=True,timeout=timeout);calls.append(text)
                if len(calls)==1:raise subprocess.TimeoutExpired('SSH-observation',timeout)
                return output
            remote.command=command;ack={'payload':{'test':'original'},'signature':'original'}
            with self.assertRaises(subprocess.TimeoutExpired):remote._durable_cache_ack(script,ack,'original')
            time.sleep(0.5)
            result=remote._durable_cache_ack(script,ack,'original')
            self.assertEqual(result['status'],'complete');self.assertEqual(counter.read_text(),'one')
            self.assertEqual(len(list(root.glob('*-attempt.json'))),1)
    def test_bounded_supervisor_failure_retains_original_handle(self):
        import tempfile
        from subnet.remote_backend import RemoteJobs
        with tempfile.TemporaryDirectory()as directory:
            remote=RemoteJobs.__new__(RemoteJobs);remote.workspace=directory;remote.python=sys.executable
            remote.command=lambda text,timeout:subprocess.check_output(['bash','-c',text],text=True,timeout=timeout)
            result=remote._durable_cache_ack('raise ValueError("private-secret-not-printed")',{'payload':{'test':'original'}},'original')
            self.assertEqual(result['reason'],'original-cache-ACK-failed')
            self.assertIn('original_handle',result)
            raw=next(Path(directory).glob('*-result.json')).read_text();self.assertNotIn('private-secret',raw)
