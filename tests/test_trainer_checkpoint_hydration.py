import base64,copy,hashlib,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from subnet.storage import Identity,canonical
from subnet.backend_jobs import file_map,signed
from ops.trainer_checkpoint_hydration import VERSION,create_plan,policy_admission

class EvaluatorHydration(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.helper=self.root/'helper.py';self.helper.write_text('scoped helper')
        self.policy=dict(version=VERSION,helper_path=str(self.helper),helper_sha256=hashlib.sha256(self.helper.read_bytes()).hexdigest(),remote_directory='/workspace/hydration',retained_UUID='node')
        key=Identity(bytes(range(32)))
        sign=lambda value:dict(payload=value,signer=key.id,signature=base64.b64encode(key.key.sign(canonical(value)).signature).decode())
        client=SimpleNamespace(head_object=Mock(return_value={'ContentLength':100}),generate_presigned_url=Mock(return_value='scoped-read'))
        self.controller=SimpleNamespace(authority=key,signed=sign,bucket=SimpleNamespace(name='bucket',client=client))
        self.remote=SimpleNamespace(workspace='/workspace/evaluator')
        files={'config.json':'a'*64,'model.safetensors':'b'*64};self.cp=dict(id=file_map(files),files=files)
    def test_plan_pins_full_inventory_read_scope_and_role_workspace(self):
        envelope=create_plan(self.controller,self.remote,self.cp,self.policy,now=100)
        plan=signed(envelope,self.controller.authority.id)
        self.assertEqual(signed(plan['checkpoint_descriptor'],self.controller.authority.id),self.cp)
        self.assertEqual(plan['destination'],self.remote.workspace+'/checkpoints/'+self.cp['id'])
        self.assertEqual(plan['expires_at'],3600);self.assertEqual(plan['GPU_runs'],0)
        self.assertEqual(plan['optimizer_runs'],0);self.assertEqual(plan['publication_writes'],0)
        self.assertEqual(set(plan['objects']),set(self.cp['files']))
        self.assertEqual(self.controller.bucket.client.generate_presigned_url.call_count,2)
        for call in self.controller.bucket.client.generate_presigned_url.call_args_list:
            self.assertEqual(call.args[0],'get_object')
            self.assertTrue(call.kwargs['Params']['Key'].startswith('public/checkpoints/'+self.cp['id']+'/'))
    def test_changed_helper_or_checkpoint_never_issues_read_capabilities(self):
        changed=copy.deepcopy(self.cp);changed['id']='0'*64
        with self.assertRaisesRegex(ValueError,'inventory'):create_plan(self.controller,self.remote,changed,self.policy)
        self.helper.write_text('changed')
        with self.assertRaisesRegex(ValueError,'helper bytes'):create_plan(self.controller,self.remote,self.cp,self.policy)
        self.controller.bucket.client.head_object.assert_not_called()
    def test_policy_bounds_helper_and_remote_path(self):
        for field,value in [('version','other'),('remote_directory','../outside'),('retained_UUID','')]:
            with self.subTest(field=field),self.assertRaises(ValueError):policy_admission(dict(self.policy,**{field:value}))



class EvaluationHydrationRenewal(EvaluatorHydration):
    def setUp(self):
        super().setUp()
        import sys,subprocess,shlex,shutil
        from unittest.mock import patch
        self.helper.write_bytes(((Path(__file__).resolve().parents[1]/'ops/checkpoint_read_hydration.py')).read_bytes())
        self.policy.update(helper_path=str(self.helper),helper_sha256=hashlib.sha256(self.helper.read_bytes()).hexdigest(),remote_directory=str(self.root/'remote-hydration'))
        self.controller.state=self.root/'state';self.controller.state.mkdir()
        self.remote.workspace=str(self.root/'evaluator');self.remote.python=sys.executable
        self.controller.bucket.client.generate_presigned_url.side_effect=lambda action,Params,ExpiresIn:'https://account.r2.cloudflarestorage.com/bucket/'+Params['Key']+'?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature='+'a'*64
        self.main_calls=0;self.adopted_sizes=[];self.command_args=[]
        def command(cmd,timeout=None):
            self.command_args.append(cmd)
            argv=shlex.split(cmd)
            if '--plan' in argv:
                self.main_calls+=1
                env=json.loads(Path(argv[argv.index('--plan')+1]).read_bytes());plan=env['payload']
                digest=argv[argv.index('--plan-sha256')+1];destination=Path(plan['destination'])
                stage=destination.parent/('.'+self.cp['id']+'.hydrate-'+digest[:16])/'objects'
                self.adopted_sizes.append((stage/'model.safetensors.partial').stat().st_size if(stage/'model.safetensors.partial').exists()else 0)
                destination.mkdir(parents=True,exist_ok=True)
                for name,meta in plan['objects'].items():(destination/name).write_bytes(b'x'*meta['bytes'])
                receipt=dict(CPU_only=True,GPU_runs=0,checkpoint=self.cp['id'],actual_files=self.cp['files'],objects=plan['objects'],actual_local_destination=str(destination),read_plan_sha256=digest,role='train-input',retained_UUID='node')
                Path(argv[argv.index('--output')+1]).write_text(json.dumps(receipt));return '{}'
            result=subprocess.run(argv,capture_output=True,text=True)
            if result.returncode:raise RuntimeError(result.stderr)
            return result.stdout
        import json
        self.remote.command=command
        self.remote.copy_to=lambda a,b:shutil.copyfile(a,b)
        self.patch=patch('ops.trainer_checkpoint_hydration.time.time',return_value=5000);self.patch.start();self.addCleanup(self.patch.stop)
    def original(self):
        import json
        from ops.trainer_checkpoint_hydration import create_plan
        env=create_plan(self.controller,self.remote,self.cp,self.policy,now=0)
        digest=hashlib.sha256(canonical(env)).hexdigest();ns=hashlib.sha256(self.remote.workspace.encode()).hexdigest()[:16]
        folder=self.controller.state/'checkpoint-hydration'/ns;folder.mkdir(parents=True)
        path=folder/(self.cp['id']+'-plan.json');path.write_bytes(canonical(env))
        return env,digest,folder,path,ns
    def test_expired_unfinished_plan_renews_and_preserves_exact_partial(self):
        import json
        from ops.trainer_checkpoint_hydration import prefetch
        env,digest,folder,path,ns=self.original();before=path.read_bytes();plan=env['payload']
        stage=Path(plan['destination']).parent/('.'+self.cp['id']+'.hydrate-'+digest[:16]);stage.mkdir(parents=True)
        (stage/'binding.json').write_text(json.dumps(dict(plan_sha256=digest,checkpoint=self.cp['id'],role='train-input',retained_UUID='node',descriptor_sha256=plan['checkpoint_descriptor_sha256'])))
        (stage/'objects').mkdir();(stage/'objects'/'model.safetensors.partial').write_bytes(b'p'*23)
        result=prefetch(self.controller,self.remote,dict(checkpoint=self.cp),self.policy)
        self.assertEqual(self.adopted_sizes,[23]);self.assertEqual(self.main_calls,1)
        self.assertFalse(any("X-Amz-"in c for c in self.command_args))
        self.assertEqual(path.read_bytes(),before);self.assertTrue((stage/'binding.json').exists())
        self.assertNotEqual(result['read_plan_sha256'],digest);self.assertEqual(len(list((folder/(self.cp['id']+'-plans')).glob('*.json'))),1)
    def test_expired_completed_receipt_does_not_refresh_or_rerun(self):
        import json
        from ops.trainer_checkpoint_hydration import prefetch
        env,digest,folder,path,ns=self.original();plan=env['payload']
        d=Path(plan['destination']);d.mkdir(parents=True)
        for name,meta in plan['objects'].items():(d/name).write_bytes(b'x'*meta['bytes'])
        r=Path(self.policy['remote_directory'])/ns/self.cp['id']/digest/'receipt.json';r.parent.mkdir(parents=True)
        receipt=dict(CPU_only=True,GPU_runs=0,checkpoint=self.cp['id'],actual_files=self.cp['files'],objects=plan['objects'],actual_local_destination=str(d),read_plan_sha256=digest,role='train-input',retained_UUID='node')
        r.write_text(json.dumps(receipt));self.controller.bucket.client.head_object.reset_mock()
        self.assertEqual(prefetch(self.controller,self.remote,dict(checkpoint=self.cp),self.policy),receipt)
        self.assertEqual(self.main_calls,0);self.controller.bucket.client.head_object.assert_not_called()
    def test_renewal_rejects_changed_actor_or_inventory(self):
        from ops.trainer_checkpoint_hydration import prefetch
        env,digest,folder,path,ns=self.original()
        for change in ('actor','filemap','size'):
            with self.subTest(change=change):
                changed=copy.deepcopy(env['payload'])
                if change=='actor':changed['retained_UUID']='different'
                elif change=='filemap':changed['checkpoint_descriptor']=self.controller.signed(dict(self.cp,files={'config.json':'c'*64}))
                else:changed['objects']['config.json']['bytes']=101
                path.write_bytes(canonical(self.controller.signed(changed)))
                with self.assertRaisesRegex(ValueError,'changed'):prefetch(self.controller,self.remote,dict(checkpoint=self.cp),self.policy)
        self.assertEqual(self.main_calls,0)
    def test_receipt_without_cache_renews_without_deleting_old_receipt(self):
        import json
        from ops.trainer_checkpoint_hydration import prefetch
        env,digest,folder,path,ns=self.original();plan=env['payload']
        r=Path(self.policy['remote_directory'])/ns/self.cp['id']/digest/'receipt.json';r.parent.mkdir(parents=True)
        receipt=dict(CPU_only=True,GPU_runs=0,checkpoint=self.cp['id'],actual_files=self.cp['files'],objects=plan['objects'],actual_local_destination=plan['destination'],read_plan_sha256=digest,role='train-input',retained_UUID='node')
        r.write_text(json.dumps(receipt));before=r.read_bytes()
        prefetch(self.controller,self.remote,dict(checkpoint=self.cp),self.policy)
        self.assertEqual(self.main_calls,1);self.assertEqual(r.read_bytes(),before)

class TypedEvaluatorTransport(unittest.TestCase):
    def test_typed_expiry_copy_does_not_change_public_helper(self):
        import importlib.util
        from ops.trainer_checkpoint_hydration import materialize_evaluator_helper
        with tempfile.TemporaryDirectory()as tmp:
            source=(Path(__file__).resolve().parents[1]/'ops/checkpoint_read_hydration.py');before=source.read_bytes();out=Path(tmp)/'helper.py'
            digest=materialize_evaluator_helper(source,out)
            self.assertEqual(source.read_bytes(),before);self.assertEqual(digest,hashlib.sha256(out.read_bytes()).hexdigest())
            module=importlib.util.spec_from_file_location('temporary_evaluator_helper',out);helper=importlib.util.module_from_spec(module);module.loader.exec_module(helper)
            with self.assertRaises(helper.HydrationReadPlanExpired):helper.download(None,'unused',Path(tmp)/'partial',100,'a'*64,0,clock=lambda:1)
            self.assertEqual(materialize_evaluator_helper(source,out),digest)
    def test_only_fresh_typed_transfer_errors_defer(self):
        from ops.trainer_checkpoint_hydration import transient_transfer
        for error in ['ReadTimeout','ConnectionError','ChunkedEncodingError','HydrationReadPlanExpired']:
            self.assertTrue(transient_transfer(dict(error_type=error,observed_at=10),9))
        for error in ['ValueError','KeyError','InvalidSignature','OSError']:
            self.assertFalse(transient_transfer(dict(error_type=error,observed_at=10),9))
        self.assertFalse(transient_transfer(dict(error_type='ReadTimeout',observed_at=8),9))

class TrainerCacheControls(unittest.TestCase):
    def case(self,cache=True,present=False,regular=False):
        import threading,json
        from unittest.mock import Mock
        d=tempfile.TemporaryDirectory();self.addCleanup(d.cleanup);root=Path(d.name)
        cp=dict(id='a'*64,files={'config.json':'b'*64});target='/workspace/trainer/checkpoints/'+cp['id']
        trainer=SimpleNamespace(workspace='/workspace/trainer',python='/bin/python',command=Mock(return_value=json.dumps(dict(present=present,regular=regular))))
        router=SimpleNamespace(roles={'train':trainer},caches={'train':{cp['id']:target}if cache else{}},owners={},controller=object(),cache_lock=threading.RLock(),cache_path=root/'caches.json',owner_path=root/'owners.json')
        return router,dict(checkpoint=cp),dict(actual_local_destination=target,actual_files=cp['files']),target
    def test_absent_recorded_copy_hydrates_then_updates_exact_cache(self):
        from unittest.mock import patch
        from ops.trainer_checkpoint_hydration import ensure_checkpoint
        r,m,receipt,target=self.case()
        with patch('ops.trainer_checkpoint_hydration.prefetch',return_value=receipt)as h:
            result=ensure_checkpoint(r,m,{})
        h.assert_called_once();self.assertEqual(result['status'],'full-SHA-hydrated');self.assertEqual(r.owners[target],'train');self.assertTrue(r.cache_path.exists())
    def test_valid_working_copy_skips_download_and_keeps_capacity_path(self):
        from unittest.mock import patch
        from ops.trainer_checkpoint_hydration import ensure_checkpoint
        r,m,receipt,target=self.case(present=True,regular=True)
        with patch('ops.trainer_checkpoint_hydration.prefetch')as h:result=ensure_checkpoint(r,m,{})
        h.assert_not_called();self.assertEqual(result['status'],'working-copy-present')
    def test_malformed_present_copy_never_overwritten(self):
        from unittest.mock import patch
        from ops.trainer_checkpoint_hydration import ensure_checkpoint
        r,m,receipt,target=self.case(present=True,regular=False)
        with patch('ops.trainer_checkpoint_hydration.prefetch')as h,self.assertRaisesRegex(ValueError,'malformed'):ensure_checkpoint(r,m,{})
        h.assert_not_called();self.assertFalse(r.cache_path.exists())
    def test_wrong_hydration_receipt_never_installs_cache(self):
        from unittest.mock import patch
        from ops.trainer_checkpoint_hydration import ensure_checkpoint
        r,m,receipt,target=self.case();receipt['actual_files']={}
        with patch('ops.trainer_checkpoint_hydration.prefetch',return_value=receipt),self.assertRaises(ValueError):ensure_checkpoint(r,m,{})
        self.assertFalse(r.cache_path.exists())
    def test_capacity_still_runs_after_automatic_hydration(self):
        from unittest.mock import patch,Mock
        from ops.trainer_checkpoint_hydration import install
        old=Mock(side_effect=ValueError('real capacity insufficient'));cl=type('Router',(),{'training_capacity':old});r=cl()
        with patch('ops.trainer_checkpoint_hydration.policy_admission'),patch('ops.trainer_checkpoint_hydration.ensure_checkpoint')as ensure:
            install(SimpleNamespace(RoutedJobs=cl),{})
            with self.assertRaisesRegex(ValueError,'real capacity insufficient'):r.training_capacity({'checkpoint':{}},1,submission_bytes=23)
        ensure.assert_called_once();old.assert_called_once_with(r,{'checkpoint':{}},1,submission_bytes=23)
if __name__=='__main__':unittest.main()
