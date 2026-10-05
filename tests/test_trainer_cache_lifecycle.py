import json,tempfile,unittest,hashlib,os,threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from subnet.storage import Identity,canonical
from subnet.trainer_cache_lifecycle import retire,VERSION
from subnet.cache_lifecycle import CacheLifecycle
from subnet.persistent_training_controller import retire_completed_cache

class TrainerRetention(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.authority=Identity()
        from subnet.remote_optimizer_readback import sign
        self.authority.sign=lambda value:sign(value,self.authority.key)
        self.jobid='original-job';self.old='a'*64;self.new='b'*64
        self.oldpath=self.root/'checkpoints'/self.old;self.oldpath.mkdir(parents=True)
        self.newpath=self.root/'jobs'/self.jobid/'checkpoint-persistent-final';self.newpath.mkdir(parents=True)
        (self.oldpath/'model.safetensors').write_bytes(b'old');(self.newpath/'model.safetensors').write_bytes(b'new')
        oldfiles={'model.safetensors':hashlib.sha256(b'old').hexdigest()};newfiles={'model.safetensors':hashlib.sha256(b'new').hexdigest()}
        self.cp={'id':self.old,'files':oldfiles};self.output={'id':self.new,'files':newfiles,'path':str(self.newpath)}
        self.pointer=dict(descriptor_sha256='c'*64,namespace='durable/original',optimizer_steps=6)
        self.job=dict(job_id=self.jobid,role='train',manifest=self.authority.sign({'checkpoint':self.cp}))
        jobhash=hashlib.sha256(canonical(self.job)).hexdigest()
        self.report=dict(job_id=self.jobid,job_sha256=jobhash,success=True,new_checkpoint=self.output,
            persistent_training_state=dict(descriptor_sha256='c'*64,namespace='durable/original',descriptor={'optimizer_steps':6}))
        (self.root/(self.jobid+'.json')).write_bytes(canonical(self.authority.sign(self.job)))
        (self.root/'jobs'/self.jobid/'report.json').write_bytes(canonical(self.report))
        (self.root/'runner-status').mkdir();self.status=self.root/'runner-status'/(self.jobid+'.json');self.status.write_text(json.dumps({'phase':'complete','exit_code':0}))
        self.value=dict(version=VERSION,job_id=self.jobid,job_sha256=jobhash,report_sha256=hashlib.sha256(canonical(self.report)).hexdigest(),input_checkpoint=self.cp,input_cache=str(self.oldpath),new_checkpoint=self.output,trainer_state=self.pointer,authority_state_committed=True)
        (self.root/'authority.seed').write_text('retain key');(self.root/'source.py').write_text('retain source')
    def run_retire(self):return retire(self.authority.sign(self.value),self.authority.id,self.root)
    def test_durable_ACK_removes_prior_input_keeps_current_and_all_evidence_without_hashing(self):
        result=self.run_retire();self.assertEqual(result['removed_checkpoints'],[self.old])
        self.assertFalse(self.oldpath.exists());self.assertTrue(self.newpath.exists())
        self.assertTrue((self.root/'jobs'/self.jobid/'report.json').exists())
        self.assertTrue((self.root/'authority.seed').exists());self.assertTrue((self.root/'source.py').exists())
        self.assertFalse(result['extra_hashing']);self.assertFalse(result['extra_R2_reads'])
        again=self.run_retire();self.assertEqual(again['removed_checkpoints'],[])
    def test_uncommitted_or_tampered_original_cannot_delete(self):
        self.value['authority_state_committed']=False
        with self.assertRaises(ValueError):self.run_retire()
        self.assertTrue(self.oldpath.exists())
        self.value['authority_state_committed']=True;self.report['success']=False
        (self.root/'jobs'/self.jobid/'report.json').write_bytes(canonical(self.report))
        with self.assertRaises(ValueError):self.run_retire()
        self.assertTrue(self.oldpath.exists())
    def test_live_original_child_defers_cleanup(self):
        ticks=Path('/proc',str(os.getpid()),'stat').read_text().rsplit(')',1)[1].split()[19]
        self.status.write_text(json.dumps(dict(phase='complete',exit_code=0,child_pid=os.getpid(),child_pid_ticks=ticks)))
        self.assertEqual(self.run_retire()['status'],'deferred');self.assertTrue(self.oldpath.exists())
    def test_inflight_checkpoint_lease_prevents_deletion(self):
        lifecycle=CacheLifecycle(self.root)
        with lifecycle.lease_checkpoint(self.old):
            with self.assertRaises(BlockingIOError):self.run_retire()
        self.assertTrue(self.oldpath.exists())
    def test_external_input_mapping_is_preserved(self):
        self.value['input_cache']=str(self.root.parent/'external-model')
        self.assertEqual(self.run_retire()['removed_checkpoints'],[]);self.assertTrue(self.oldpath.exists())
    def test_newer_adopted_state_prevents_stale_cleanup(self):
        self.run_retire();marker=self.root/'.cache-lifecycle/trainer-current-state.json'
        marker.write_text(json.dumps(dict(optimizer_steps=7,descriptor_sha256='d'*64)))
        self.assertEqual(self.run_retire()['status'],'superseded');self.assertTrue(self.newpath.exists())
    def test_receipted_downloads_removed_unreceipted_diagnostics_retained(self):
        downloaded=self.root/'jobs'/self.jobid/'submission-0.json';downloaded.write_bytes(b'input')
        diagnostics=self.root/'jobs'/self.jobid/'other.json';diagnostics.write_text('retain')
        CacheLifecycle(self.root).record_download(downloaded,hashlib.sha256(b'input').hexdigest())
        self.run_retire();self.assertFalse(downloaded.exists());self.assertTrue(diagnostics.exists())
    def test_cleanup_timeout_does_not_block_or_invalidate_epoch(self):
        entered=threading.Event();release=threading.Event()
        def action(*args):entered.set();release.wait(3);raise TimeoutError('private unreachable endpoint')
        controller=SimpleNamespace(state=self.root,jobs=SimpleNamespace(retire_training_cache=action))
        (self.root/'roles').mkdir()
        thread=retire_completed_cache(controller,self.job,self.report,self.pointer)
        self.assertTrue(entered.wait(1));self.assertTrue(thread.is_alive())
        release.set();thread.join(2)
        result=json.loads((self.root/'roles'/(self.jobid+'-trainer-cache-cleanup.json')).read_text())
        self.assertEqual(result,dict(status='deferred',reason='TimeoutError',removed_checkpoints=[]))

    def test_actual_CPU_overlay_remote_glue_authenticates_then_retires_owned_paths(self):
        import subprocess,sys
        from subnet.remote_backend import RemoteJobs
        code=Path(__file__).resolve().parents[1]
        names=('subnet/cache_lifecycle.py','subnet/trainer_cache_lifecycle.py')
        files={n:hashlib.sha256((code/n).read_bytes()).hexdigest() for n in names}
        remote=RemoteJobs.__new__(RemoteJobs);remote.code=str(code);remote.workspace=str(self.root)
        remote.python=sys.executable;remote.config={'cache_lifecycle_overlay':{'code':str(code),'files':files}}
        remote.controller=SimpleNamespace(authority=self.authority,signed=self.authority.sign)
        remote.command=lambda text,timeout:subprocess.check_output(['bash','-c',text],text=True,timeout=timeout)
        result=remote.retire_training_cache(self.job,self.report,self.pointer,str(self.oldpath))
        self.assertEqual(result['removed_checkpoints'],[self.old]);self.assertTrue(self.newpath.exists())
        self.assertTrue((self.root/'authority.seed').exists())
    def test_router_prunes_only_actual_removed_trainer_paths(self):
        from subnet.role_router import RoutedJobs
        router=RoutedJobs.__new__(RoutedJobs);router.cache_lock=threading.RLock()
        router.controller=SimpleNamespace(authority=self.authority)
        router.caches={'train':{self.old:str(self.oldpath),self.new:str(self.newpath)},'mine':{self.old:'/external/miner'}}
        router.owners={str(self.oldpath):'train',str(self.newpath):'train','/external/miner':'mine'}
        router.cache_path=self.root/'cache-mappings.json';router.owner_path=self.root/'owners.json'
        action=Mock(return_value={'status':'complete','removed_checkpoints':[self.old]})
        router.roles={'train':SimpleNamespace(retire_training_cache=action)}
        router.retire_training_cache(self.job,self.report,self.pointer)
        self.assertEqual(router.caches['train'],{self.new:str(self.newpath)})
        self.assertEqual(router.caches['mine'],{self.old:'/external/miner'})
        self.assertEqual(router.owners,{str(self.newpath):'train','/external/miner':'mine'})
