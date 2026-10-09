import json,tempfile,unittest,hashlib,os,threading,copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
from subnet.storage import Identity,canonical
from subnet.trainer_cache_lifecycle import retire,VERSION,MARKER_VERSION,TRANSITION_VERSION,digest,migrate_legacy_marker
from nacl.exceptions import BadSignatureError
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
        self.genesis='e'*64
        descriptor=dict(optimizer_steps=6,genesis_sha256=self.genesis,inference_checkpoint=self.new)
        self.pointer=dict(descriptor_sha256=digest(descriptor),namespace='durable/original',optimizer_steps=6,genesis_sha256=self.genesis)
        self.job=dict(job_id=self.jobid,role='train',manifest=self.authority.sign({'checkpoint':self.cp,'trainer_state_binding':{'genesis_sha256':self.genesis}}))
        jobhash=hashlib.sha256(canonical(self.job)).hexdigest()
        self.report=dict(job_id=self.jobid,job_sha256=jobhash,success=True,new_checkpoint=self.output,
            persistent_training_state=dict(descriptor_sha256=digest(descriptor),namespace='durable/original',descriptor=descriptor))
        (self.root/(self.jobid+'.json')).write_bytes(canonical(self.authority.sign(self.job)))
        (self.root/'jobs'/self.jobid/'report.json').write_bytes(canonical(self.report))
        (self.root/'runner-status').mkdir();self.status=self.root/'runner-status'/(self.jobid+'.json');self.status.write_text(json.dumps({'phase':'complete','exit_code':0}))
        self.value=dict(version=VERSION,job_id=self.jobid,job_sha256=jobhash,report_sha256=hashlib.sha256(canonical(self.report)).hexdigest(),input_checkpoint=self.cp,input_cache=str(self.oldpath),new_checkpoint=self.output,trainer_state=self.pointer,authority_state_committed=True)
        (self.root/'authority.seed').write_text('retain key');(self.root/'source.py').write_text('retain source')
    def run_retire(self):return retire(self.authority.sign(self.value),self.authority.id,self.root)
    def other_ack(self,jobid,genesis,step,*,local_cache=False):
        job=copy.deepcopy(self.job);job['job_id']=jobid
        manifest={'checkpoint':self.cp,'trainer_state_binding':{'genesis_sha256':genesis}}
        if local_cache:manifest['optimizer_state_local_cache']={'version':'test-explicit-cache'}
        job['manifest']=self.authority.sign(manifest);jobhash=digest(job)
        descriptor=dict(optimizer_steps=step,genesis_sha256=genesis,inference_checkpoint=self.new)
        pointer=dict(self.pointer,optimizer_steps=step,genesis_sha256=genesis,descriptor_sha256=digest(descriptor))
        report=dict(self.report,job_id=jobid,job_sha256=jobhash,persistent_training_state=dict(descriptor_sha256=digest(descriptor),namespace=pointer['namespace'],descriptor=descriptor))
        (self.root/(jobid+'.json')).write_bytes(canonical(self.authority.sign(job)))
        directory=self.root/'jobs'/jobid;directory.mkdir(exist_ok=True)
        (directory/'report.json').write_bytes(canonical(report))
        (self.root/'runner-status'/(jobid+'.json')).write_text(json.dumps({'phase':'complete','exit_code':0}))
        return self.authority.sign(dict(self.value,job_id=jobid,job_sha256=jobhash,report_sha256=digest(report),trainer_state=pointer))
    def transition(self,previous,previous_ack,next_ack):
        body=dict(version=TRANSITION_VERSION,workspace=str(self.root),previous_marker_sha256=digest(previous),previous_ack=previous_ack,next_ack_sha256=digest(next_ack),from_genesis_sha256=previous_ack['payload']['trainer_state']['genesis_sha256'],to_genesis_sha256=next_ack['payload']['trainer_state']['genesis_sha256'])
        path=self.root/'.cache-lifecycle/trainer-run-transition.json';path.parent.mkdir(exist_ok=True)
        envelope=self.authority.sign(body);path.write_bytes(canonical(envelope));return envelope
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
        self.run_retire();future=self.other_ack('future-job',self.genesis,7)
        retire(future,self.authority.id,self.root)
        self.assertEqual(self.run_retire()['status'],'superseded');self.assertTrue(self.newpath.exists())

    def test_new_marker_genesis_derived_from_original_ack(self):
        self.run_retire();marker=json.loads((self.root/'.cache-lifecycle/trainer-current-state.json').read_text())
        self.assertEqual(marker['version'],MARKER_VERSION);self.assertEqual(marker['genesis_sha256'],self.genesis)
        self.assertEqual(marker['ROOT_ack'],self.authority.sign(self.value));self.assertEqual(marker['retired_geneses'],[])

    def test_descriptor_genesis_must_match_pointer_and_original_manifest(self):
        for field in ('pointer','manifest','descriptor_hash'):
            ack=self.other_ack('bad-'+field,self.genesis,7)
            value=ack['payload']
            if field=='pointer':value['trainer_state']['genesis_sha256']='f'*64
            elif field=='manifest':
                path=self.root/(value['job_id']+'.json');job=json.loads(path.read_text())['payload']
                manifest=job['manifest']['payload'];manifest['trainer_state_binding']['genesis_sha256']='f'*64
                job['manifest']=self.authority.sign(manifest);path.write_bytes(canonical(self.authority.sign(job)))
                value['job_sha256']=digest(job)
                reportpath=self.root/'jobs'/value['job_id']/'report.json';report=json.loads(reportpath.read_text())
                report['job_sha256']=value['job_sha256'];reportpath.write_bytes(canonical(report));value['report_sha256']=digest(report)
            else:
                reportpath=self.root/'jobs'/value['job_id']/'report.json';report=json.loads(reportpath.read_text())
                report['persistent_training_state']['descriptor']['extra']='changed-without-descriptor-hash'
                reportpath.write_bytes(canonical(report));value['report_sha256']=digest(report)
            with patch('subnet.trainer_cache_lifecycle._retire_owned')as cleanup:
                with self.assertRaisesRegex(ValueError,'lineage binding'):retire(self.authority.sign(value),self.authority.id,self.root)
                cleanup.assert_not_called()

    def test_legacy_exact_descriptor_upgrades_without_counter_guess(self):
        marker=self.root/'.cache-lifecycle/trainer-current-state.json';marker.parent.mkdir()
        marker.write_text(json.dumps({k:self.pointer[k]for k in ('optimizer_steps','descriptor_sha256')}))
        self.assertEqual(self.run_retire()['status'],'complete')
        self.assertEqual(json.loads(marker.read_text())['genesis_sha256'],self.genesis)

    def test_unknown_legacy_higher_counter_preserves_everything(self):
        marker=self.root/'.cache-lifecycle/trainer-current-state.json';marker.parent.mkdir()
        original=dict(optimizer_steps=99,descriptor_sha256='d'*64);marker.write_text(json.dumps(original))
        with self.assertRaises(FileNotFoundError):self.run_retire()
        self.assertEqual(json.loads(marker.read_text()),original);self.assertTrue(self.oldpath.exists())

    def test_explicit_new_genesis_step_one_and_delayed_old_ack(self):
        oldack=self.authority.sign(self.value);self.run_retire()
        marker=self.root/'.cache-lifecycle/trainer-current-state.json';previous=json.loads(marker.read_text())
        newack=self.other_ack('new-run-step-one','f'*64,1)
        self.transition(previous,oldack,newack)
        self.assertEqual(retire(newack,self.authority.id,self.root)['status'],'complete')
        current=marker.read_bytes();saved=json.loads(current)
        self.assertEqual(saved['optimizer_steps'],1);self.assertEqual(saved['genesis_sha256'],'f'*64)
        self.assertEqual(saved['retired_geneses'],[self.genesis])
        self.assertEqual(retire(oldack,self.authority.id,self.root)['reason'],'retired-genesis')
        self.assertEqual(marker.read_bytes(),current)
        self.assertEqual(retire(newack,self.authority.id,self.root)['status'],'complete')

    def test_legacy_transition_requires_authentic_original_previous_report(self):
        oldack=self.authority.sign(self.value)
        marker=self.root/'.cache-lifecycle/trainer-current-state.json';marker.parent.mkdir()
        previous={k:self.pointer[k]for k in ('optimizer_steps','descriptor_sha256')};marker.write_text(json.dumps(previous))
        newack=self.other_ack('new-run-step-one','f'*64,1);self.transition(previous,oldack,newack)
        self.assertEqual(retire(newack,self.authority.id,self.root)['status'],'complete')
        self.assertEqual(json.loads(marker.read_text())['retired_geneses'],[self.genesis])

    def test_foreign_genesis_without_transition_is_not_numeric_supersession(self):
        self.run_retire();marker=self.root/'.cache-lifecycle/trainer-current-state.json';before=marker.read_bytes()
        newack=self.other_ack('new-run-step-one','f'*64,1)
        with self.assertRaises(FileNotFoundError):retire(newack,self.authority.id,self.root)
        self.assertEqual(marker.read_bytes(),before)

    def test_transition_wrong_marker_or_ack_or_signature_never_deletes(self):
        self.run_retire();marker=self.root/'.cache-lifecycle/trainer-current-state.json';previous=json.loads(marker.read_text())
        newack=self.other_ack('new-run-step-one','f'*64,1);oldack=self.authority.sign(self.value)
        for field,value in [('previous_marker_sha256','0'*64),('next_ack_sha256','0'*64),('to_genesis_sha256',self.genesis)]:
            env=self.transition(previous,oldack,newack);env['payload'][field]=value
            (marker.parent/'trainer-run-transition.json').write_bytes(canonical(self.authority.sign(env['payload'])))
            with patch('subnet.trainer_cache_lifecycle._retire_owned')as cleanup:
                with self.assertRaises(ValueError):retire(newack,self.authority.id,self.root)
                cleanup.assert_not_called()
        env=self.transition(previous,oldack,newack);env['payload']['workspace']='tampered'
        (marker.parent/'trainer-run-transition.json').write_bytes(canonical(env))
        with self.assertRaises(BadSignatureError):retire(newack,self.authority.id,self.root)

    def test_same_genesis_same_counter_different_descriptor_refused(self):
        self.run_retire();newack=self.other_ack('conflicting-job',self.genesis,6)
        reportpath=self.root/'jobs/conflicting-job/report.json';report=json.loads(reportpath.read_text())
        descriptor=report['persistent_training_state']['descriptor'];descriptor['extra']='different-state'
        report['persistent_training_state']['descriptor_sha256']=digest(descriptor);reportpath.write_bytes(canonical(report))
        value=newack['payload'];value['report_sha256']=digest(report);value['trainer_state']['descriptor_sha256']=digest(descriptor)
        with self.assertRaisesRegex(ValueError,'same counter'):retire(self.authority.sign(value),self.authority.id,self.root)

    def test_promotion_failure_keeps_previous_marker_and_checkpoint(self):
        ack=self.other_ack('cache-job',self.genesis,7,local_cache=True)
        marker=self.root/'.cache-lifecycle/trainer-current-state.json'
        with patch('subnet.optimizer_state_cache.promote',side_effect=ValueError('promotion failed')):
            with self.assertRaisesRegex(ValueError,'promotion failed'):retire(ack,self.authority.id,self.root)
        self.assertFalse(marker.exists());self.assertTrue(self.oldpath.exists())
        with patch('subnet.optimizer_state_cache.promote',return_value={'promoted':True}),patch('subnet.trainer_cache_lifecycle._retire_owned',side_effect=OSError('interrupted cleanup')):
            with self.assertRaises(OSError):retire(ack,self.authority.id,self.root)
        self.assertFalse(marker.exists())
        with patch('subnet.optimizer_state_cache.promote',return_value={'promoted':True,'idempotent':True}):
            self.assertEqual(retire(ack,self.authority.id,self.root)['status'],'complete')

    def test_exact_legacy_migration_then_ordinary_next_step(self):
        marker=self.root/'.cache-lifecycle/trainer-current-state.json';marker.parent.mkdir()
        marker.write_text(json.dumps({k:self.pointer[k]for k in ('optimizer_steps','descriptor_sha256')}))
        ack=self.authority.sign(self.value)
        self.assertFalse(migrate_legacy_marker(ack,self.authority.id,self.root)['idempotent'])
        self.assertTrue(migrate_legacy_marker(ack,self.authority.id,self.root)['idempotent'])
        self.assertTrue(self.oldpath.exists())
        nextack=self.other_ack('next-job',self.genesis,7)
        self.assertEqual(retire(nextack,self.authority.id,self.root)['status'],'complete')
        before=marker.read_bytes()
        with self.assertRaises(ValueError):migrate_legacy_marker(ack,self.authority.id,self.root)
        self.assertEqual(marker.read_bytes(),before)

    def test_new_promotion_before_marker_blocks_delayed_old_uncached_ack(self):
        oldack=self.authority.sign(self.value);self.run_retire()
        marker=self.root/'.cache-lifecycle/trainer-current-state.json';before=marker.read_bytes()
        newack=self.other_ack('new-run-step-one','f'*64,1)
        self.transition(json.loads(before),oldack,newack)
        cache=self.root/'.optimizer-state-cache';cache.mkdir()
        value=newack['payload']
        (cache/'current.json').write_bytes(canonical(dict(ROOT_ack=newack,descriptor_sha256=value['trainer_state']['descriptor_sha256'],job_id=value['job_id'],job_sha256=value['job_sha256'])))
        with patch('subnet.trainer_cache_lifecycle._retire_owned')as cleanup:
            with self.assertRaisesRegex(ValueError,'promoted optimizer genesis'):retire(oldack,self.authority.id,self.root)
            cleanup.assert_not_called()
        self.assertEqual(marker.read_bytes(),before)
        self.assertEqual(retire(newack,self.authority.id,self.root)['status'],'complete')
    def test_receipted_downloads_removed_unreceipted_diagnostics_retained(self):
        downloaded=self.root/'jobs'/self.jobid/'submission-0.json';downloaded.write_bytes(b'input')
        diagnostics=self.root/'jobs'/self.jobid/'other.json';diagnostics.write_text('retain')
        CacheLifecycle(self.root).record_download(downloaded,hashlib.sha256(b'input').hexdigest())
        self.run_retire();self.assertFalse(downloaded.exists());self.assertTrue(diagnostics.exists())
    def test_cleanup_timeout_does_not_block_or_invalidate_epoch(self):
        entered=threading.Event();release=threading.Event()
        def action(*args):entered.set();release.wait(3);raise TimeoutError('private unreachable endpoint')
        controller=SimpleNamespace(state=self.root,authority=self.authority,jobs=SimpleNamespace(retire_training_cache=action))
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
