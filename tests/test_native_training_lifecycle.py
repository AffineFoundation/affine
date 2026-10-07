import copy
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import test_native_training_eligibility as integration
from ops.native_training_eligibility import NativeNoUpdate,_canonical,_load
from ops.native_training_lifecycle import close_no_update,completion_fields,retire_completed,document_path,install_document_bundle

class MemoryBucket:
    def __init__(self):self.objects={};self.gets=[];self.fail=False
    def get_bounded(self,key,*,limit):
        self.gets.append(key)
        if self.fail:raise ConnectionError('fixture transient')
        return self.objects[key]
    def put(self,key,data):self.objects[key]=data
    def json(self,key,value):self.put(key,_canonical(value))

class NativeLifecycle(unittest.TestCase):
    setUp=integration.SelectorIntegration.setUp
    grade=integration.SelectorIntegration.grade
    select=integration.SelectorIntegration.select
    assert_originals=integration.SelectorIntegration.assert_originals
    def prepare(self,indeterminate=False):
        pointer=dict(optimizer_steps=24,inference_checkpoint=self.manifest['checkpoint']['id'],parent='unchanged-state')
        self.manifest['trainer_state_binding']=dict(global_step_before=24,parent=pointer)
        self.manifest['continuous_reward_contract']={'version':'CPU-ONLY'};self.manifest['max_batches']=3
        self.population.write_bytes(_canonical(dict(version='committed-unaudited-training-v1',manifest=self.manifest,submissions=self.submissions,population={'assurance':'unaudited'})))
        self.original_files={p:p.read_bytes()for p in (self.population,self.selection)}
        self.statuses=['excluded_indeterminate' if indeterminate else 'excluded_label_mismatch']*2
        self.controller.bucket=MemoryBucket()
        try:self.select()
        except NativeNoUpdate:pass
        self.status=dict(checkpoint=copy.deepcopy(self.manifest['checkpoint']),checkpoint_path='/approved/existing/model',trainer_state=pointer,training_steps=24,persistent_state_committed=True,public_optimizer_steps=24)
        self.latest=self.state/'latest-trainer-state.json';self.latest.write_bytes(_canonical(pointer));self.pointer_bytes=self.latest.read_bytes()
        return self.state/'native-outcome-eligibility'/self.epoch
    def complete(self):
        cp,metrics=close_no_update(self.controller,NativeNoUpdate(),self.manifest,self.status)
        value=dict(epoch=self.epoch,round=35,checkpoint=cp['id'],next_checkpoint=cp['id'],input_assurance='unaudited',**completion_fields(self.controller,self.epoch))
        (self.state/(self.epoch+'-signed-learner-completion.json')).write_bytes(_canonical(self.sign(value)))
        (self.state/'controller.json').write_bytes(_canonical(dict(self.status,round=36,active=None,last_completed_epoch=value)))
        return metrics
    def captured(self):
        from subnet.distributed_roles import authenticate
        # Add authentic original captured-key identity to test admissions/context,
        # preserving original fixture data, and then rebuild selector evidence.
        self.submissions=[]
        for i in range(2):
            data=_canonical({'fixture':i});source=self.state/('original-'+str(i));source.write_bytes(data)
            import hashlib
            sha=hashlib.sha256(data).hexdigest()
            admission=dict(epoch=self.epoch,document_sha256=sha,document_size=len(data),miner_identity=str(i)*64,commitment_sha256='d'*64,slot=i)
            self.submissions.append(dict(sha256=sha,size=len(data),url=source.as_uri(),learner_admission=self.sign(admission)))
    def add_original_r2(self):
        for obj in self.submissions:
            a=obj['learner_admission']['payload'];key='public/'+self.epoch+'/submissions/'+a['miner_identity']+'/'+a['commitment_sha256']+'/training/'+str(a['slot'])+'.json'
            data=_load(document_path(self.state/'native-outcome-eligibility'/self.epoch,obj['sha256']))
            self.controller.bucket.objects[key]=data
    def test_no_update_preserves_pointer_no_job_signed_reason_and_restart(self):
        self.prepare();m=self.complete();self.assertEqual(m['steps'],0);self.assertFalse(m['native_no_update']['training_dispatched'])
        first=_load(self.state/'native-outcome-eligibility'/self.epoch/'no-update.ROOT-SIGNED.json')
        self.complete();self.assertEqual(first,_load(self.state/'native-outcome-eligibility'/self.epoch/'no-update.ROOT-SIGNED.json'))
        self.assertEqual(self.latest.read_bytes(),self.pointer_bytes);self.assert_originals();self.assertFalse((self.state/'roles').exists())
    def test_indeterminate_truthful_new_epoch_retry_not_learning_or_fraud(self):
        self.prepare(True);m=self.complete()['native_no_update']
        self.assertEqual(m['status'],'closed_native_indeterminate_no_update');self.assertEqual(m['indeterminate_pairs'],2)
        self.assertEqual(m['retry_policy'],'new_unopened_epoch_same_parent_only');self.assertFalse(m['cheating_penalties']);self.assertEqual(m['optimizer_updates'],0)
    def test_other_exception_or_changed_pointer_or_issued_job_refuse(self):
        self.prepare()
        with self.assertRaises(ConnectionError):close_no_update(self.controller,ConnectionError(),self.manifest,self.status)
        self.latest.write_text('{}')
        with self.assertRaises(ValueError):self.complete()
        self.latest.write_bytes(self.pointer_bytes);(self.state/'roles').mkdir();(self.state/'roles'/(self.epoch+'-train.json')).write_text('original')
        with self.assertRaises(ValueError):self.complete()
    def test_missing_completion_or_failed_full_get_never_unlinks(self):
        self.captured();root=self.prepare()
        self.assertEqual(retire_completed(self.controller,self.epoch)['retired_bytes'],0)
        self.complete();self.add_original_r2();self.controller.bucket.fail=True
        with self.assertRaises(ConnectionError):retire_completed(self.controller,self.epoch)
        self.assertEqual(sum(document_path(root,o['sha256']).exists() for o in self.submissions),2);self.assertFalse((root/'full-R2-ACK.ROOT-SIGNED.json').exists())
    def test_full_r2_ack_then_owned_copy_retirement_only_replay_safe(self):
        self.captured();root=self.prepare();self.complete();self.add_original_r2()
        originals={k:v for k,v in self.controller.bucket.objects.items()if k.startswith('public/')}
        first=retire_completed(self.controller,self.epoch,max_documents=1);self.assertEqual(first['status'],'full_readback_in_progress');self.assertEqual(sum(document_path(root,o['sha256']).exists() for o in self.submissions),2)
        result=retire_completed(self.controller,self.epoch);self.assertEqual(result['retired_files'],2);self.assertEqual(sum(document_path(root,o['sha256']).exists() for o in self.submissions),0)
        self.assertTrue((root/'full-R2-ACK.ROOT-SIGNED.json').exists());self.assertEqual(retire_completed(self.controller,self.epoch)['retired_files'],0)
        self.assert_originals();self.assertEqual(self.latest.read_bytes(),self.pointer_bytes)
        for key,data in originals.items():self.assertEqual(self.controller.bucket.objects[key],data)
    def test_inode_replacement_foreign_copy_and_forged_ack_refuse(self):
        self.captured();root=self.prepare();self.complete();self.add_original_r2()
        target=document_path(root,self.submissions[0]['sha256']);data=target.read_bytes();target.unlink();target.write_bytes(data)
        with self.assertRaises(ValueError):retire_completed(self.controller,self.epoch)
        self.assertTrue(target.exists());self.assert_originals()
    def test_active_selector_lease_prevents_any_cleanup_or_get(self):
        import fcntl,os
        self.captured();root=self.prepare();self.complete();self.add_original_r2()
        fd=os.open(root/'selector.lock',os.O_RDWR);fcntl.flock(fd,fcntl.LOCK_EX)
        try:
            self.assertEqual(retire_completed(self.controller,self.epoch)['status'],'deferred_active_selector_lease')
            self.assertEqual(self.controller.bucket.gets,[])
        finally:os.close(fd)
    def test_authenticated_but_wrong_ack_member_refuses(self):
        self.captured();root=self.prepare();self.complete();self.add_original_r2()
        # Interrupt after authentic full ACK, before any unlink, then alter the
        # acknowledged membership under a valid test signature.
        target=document_path(root,self.submissions[0]['sha256']);data=target.read_bytes();target.unlink();target.write_bytes(data)
        with self.assertRaises(ValueError):retire_completed(self.controller,self.epoch)
        path=root/'full-R2-ACK.ROOT-SIGNED.json';ack=json.loads(path.read_text())['payload'];ack['documents'][0]['sha256']='f'*64
        path.write_bytes(_canonical(self.sign(ack)))
        with self.assertRaisesRegex(ValueError,'exact original'):retire_completed(self.controller,self.epoch)
        self.assertTrue(target.exists())
    def test_crash_after_unlink_recovers_same_intent_without_adoption(self):
        import ops.native_training_lifecycle as module
        self.captured();root=self.prepare();self.complete();self.add_original_r2()
        create=module._create;failed=[False]
        def crash(path,value):
            if str(path).endswith('.retired.ROOT-SIGNED.json')and not failed[0]:failed[0]=True;raise RuntimeError('simulated crash after unlink')
            return create(path,value)
        with patch.object(module,'_create',side_effect=crash):
            with self.assertRaises(RuntimeError):retire_completed(self.controller,self.epoch)
        self.assertTrue(failed[0]);self.assertEqual(retire_completed(self.controller,self.epoch)['status'],'owned_documents_retired')
        self.assertEqual(sum(document_path(root,o['sha256']).exists() for o in self.submissions),0)
        self.assert_originals()
    def test_completion_signature_and_no_update_binding_refuse(self):
        self.captured();root=self.prepare();self.complete();self.add_original_r2()
        path=self.state/(self.epoch+'-signed-learner-completion.json');e=json.loads(path.read_text());e['payload']['optimizer_updates']=1
        path.write_bytes(_canonical(e))
        with self.assertRaises(Exception):retire_completed(self.controller,self.epoch)
        path.write_bytes(_canonical(self.sign(e['payload'])))
        with self.assertRaises(ValueError):retire_completed(self.controller,self.epoch)
        self.assertTrue((document_path(root,self.submissions[0]['sha256'])).exists())
    def test_crash_before_unlink_reuses_exact_intent_and_same_owned_inode(self):
        self.captured();root=self.prepare();self.complete();self.add_original_r2()
        target=document_path(root,self.submissions[0]['sha256']);original_unlink=Path.unlink
        failed=[False]
        def crash(path,*args,**kwargs):
            if path==target and not failed[0]:failed[0]=True;raise RuntimeError('before unlink')
            return original_unlink(path,*args,**kwargs)
        with patch.object(Path,'unlink',crash):
            with self.assertRaises(RuntimeError):retire_completed(self.controller,self.epoch)
        self.assertTrue(target.exists());self.assertTrue((root/('document-'+self.submissions[0]['sha256']+'.json.retirement-intent.ROOT-SIGNED.json')).exists())
        self.assertEqual(retire_completed(self.controller,self.epoch)['retired_files'],2);self.assert_originals()
    def test_crash_before_ownership_never_exposes_or_adopts_orphan_copy(self):
        import ops.native_training_lifecycle as module
        import hashlib
        root=self.state/'pair-installation';root.mkdir();data=_canonical({'genuine':'unchanged'});sha=hashlib.sha256(data).hexdigest();create=module._create
        def crash(path,value):
            if path.name=='ownership.ROOT-SIGNED.json':raise RuntimeError('before ownership')
            return create(path,value)
        with patch.object(module,'_create',side_effect=crash):
            with self.assertRaises(RuntimeError):install_document_bundle(self.controller,root,sha,data)
        self.assertFalse(document_path(root,sha).exists());orphans=list(root.glob('.document-*.install-*'));self.assertEqual(len(orphans),1)
        orphan=(orphans[0]/'document.json').read_bytes()
        install_document_bundle(self.controller,root,sha,data)
        self.assertEqual(document_path(root,sha).read_bytes(),data);self.assertEqual((orphans[0]/'document.json').read_bytes(),orphan)
        self.assertNotEqual(document_path(root,sha).stat().st_ino,(orphans[0]/'document.json').stat().st_ino)
    def test_signed_completion_while_current_epoch_active_defers_retirement(self):
        self.captured();root=self.prepare();self.complete();self.add_original_r2()
        (self.state/'controller.json').write_bytes(_canonical(dict(self.status,round=35,active={'epoch':self.epoch})))
        result=retire_completed(self.controller,self.epoch);self.assertEqual(result['status'],'deferred_epoch_not_fully_closed')
        self.assertEqual(self.controller.bucket.gets,[]);self.assertTrue(document_path(root,self.submissions[0]['sha256']).exists())
    def test_actual_gpu_service_no_update_once_loop_closes_same_parent(self):
        self.prepare(True)
        self.status.update(active=dict(epoch=self.epoch,phase='train',started_at=1,phase_started_at=1),round=35,initial_published=True)
        (self.state/'controller.json').write_bytes(_canonical(self.status));(self.state/(self.epoch+'-manifest.json')).write_bytes(_canonical(self.manifest))
        path=Path(__file__).parents[1]/'prospective/native-training-controller-overlay/subnet/gpu_service.py'
        spec=importlib.util.spec_from_file_location('subnet.native_no_update_service_test',path);service=importlib.util.module_from_spec(spec);spec.loader.exec_module(service)
        def train(*a,**kw):raise NativeNoUpdate('empty authenticated subset')
        self.controller.train=train;self.controller.handle_native_no_update=lambda e,m,s:close_no_update(self.controller,e,m,s)
        self.controller.native_completion_fields=lambda e:completion_fields(self.controller,e)
        config=dict(epoch_prefix='nonpayable-CPU-ONLY',state=str(self.state),bucket={},remote={},duration=600,max_batches=3)
        with patch.object(service,'Bucket',return_value=self.controller.bucket),patch.object(service,'Gateway',return_value=None),patch.object(service,'RemoteController',return_value=self.controller),patch.object(service,'ChainAdapter',return_value=None),patch.object(service,'epoch_policy'),patch.object(service,'evaluate'),patch.object(service,'epoch_completion',return_value=dict(controller_completed_at=2)):
            service.run(config,once=True)
        status=json.loads((self.state/'controller.json').read_text());self.assertIsNone(status['active']);self.assertEqual(status['round'],36)
        self.assertEqual(status['trainer_state'],self.status['trainer_state']);self.assertEqual(status['training_steps'],24);self.assertEqual(status['checkpoint'],self.status['checkpoint'])
        completion=json.loads((self.state/(self.epoch+'-signed-learner-completion.json')).read_text())['payload']
        self.assertEqual(completion['optimizer_updates'],0);self.assertEqual(completion['status'],'closed_native_indeterminate_no_update')
        self.assertEqual(self.latest.read_bytes(),self.pointer_bytes);self.assertFalse((self.state/'roles').exists());self.assert_originals()

if __name__=='__main__':unittest.main()
