"""Authority-last and bounded-overlap controls; scientific validator is separate."""
import contextlib,io,json,os,tempfile,threading,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
from nacl.signing import SigningKey
from subnet.persistent_publication import complete,validate_policy,VERSION
from subnet import remote_optimizer_readback as r
from subnet.independent_state_dispatch import IndependentStateReader,launch_code,write_once,ADMISSION

class PublicationControls(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);(self.root/'roles').mkdir()
        self.key=SigningKey.generate();self.authority=bytes(self.key.verify_key).hex()
        self.policy=dict(version=VERSION,state_readback='local-full',checkpoint_readback_workers=4)
        self.manifest={'persistent_publication_policy':self.policy}
        self.job=dict(job_id='original',manifest=r.sign(self.manifest,self.key))
        self.report={'new_checkpoint':dict(id='a'*64,files={'model.safetensors':'b'*64},path='/original')}
        self.envelope=r.sign(self.job,self.key)
        (self.root/'roles/original-job.json').write_bytes(r.canonical(self.envelope))
        self.controller=SimpleNamespace(state=self.root,authority=SimpleNamespace(id=self.authority),
            publish_remote_checkpoint=Mock(),independent_state_reader=None)
    def run_overlap(self,remote=False,fail=None):
        barrier=threading.Barrier(2,timeout=3);finished=set();lock=threading.Lock()
        def operation(kind):
            barrier.wait()
            with lock:finished.add(kind)
            if fail==kind:raise ValueError('original '+kind+' failure')
            return {} if remote and kind=='state' else ({'shards':[]},'original-namespace')if kind=='state'else {'id':'checkpoint'}
        self.controller.publish_remote_checkpoint.side_effect=lambda *_:operation('checkpoint')
        if remote:
            self.policy['state_readback']='qualified-remote-full'
            self.job['manifest']=r.sign(self.manifest,self.key);self.envelope=r.sign(self.job,self.key)
            (self.root/'roles/original-job.json').write_bytes(r.canonical(self.envelope))
            self.controller.independent_state_reader=SimpleNamespace(prepare_original_readback=Mock(side_effect=lambda *_:operation('state')))
        def commit(*args,**kwargs):
            self.assertEqual(finished,{'state','checkpoint'})
            return {'optimizer_steps':4}
        with patch('subnet.persistent_training_protocol.validate_report'), \
             patch('subnet.persistent_training_protocol.independently_verify',side_effect=lambda *_:operation('state')), \
             patch('subnet.persistent_training_protocol._publish_verified_descriptor',side_effect=commit)as local, \
             patch('subnet.remote_state_commit.independently_commit_remote',side_effect=commit)as remote_commit:
            self.local=local;self.remote=remote_commit
            return complete(self.controller,self.report,self.job,self.manifest,'/original')
    def test_both_paths_overlap_before_state_authority(self):
        checkpoint,pointer,timing=self.run_overlap()
        self.assertEqual(pointer['optimizer_steps'],4);self.assertTrue(timing['authority_state_signed_after_checkpoint'])
        self.local.assert_called_once();self.remote.assert_not_called()
    def test_qualified_reader_overlaps_without_Arbos_full_state_stream(self):
        _,pointer,_=self.run_overlap(remote=True)
        self.assertEqual(pointer['optimizer_steps'],4);self.remote.assert_called_once();self.local.assert_not_called()
        self.assertEqual(self.controller.independent_state_reader.prepare_original_readback.call_args.args[2],self.envelope)
    def test_either_path_failure_never_signs_optimizer_authority(self):
        for failure in ('state','checkpoint'):
            with self.subTest(failure=failure),self.assertRaises(ValueError):self.run_overlap(fail=failure)
            self.local.assert_not_called();self.remote.assert_not_called()
    def test_unqualified_remote_never_silently_falls_back(self):
        self.policy['state_readback']='qualified-remote-full'
        with patch('subnet.persistent_training_protocol.validate_report'),self.assertRaises(ValueError):complete(self.controller,self.report,self.job,self.manifest,'/original')
        self.controller.publish_remote_checkpoint.assert_not_called()
    def test_original_serial_path_remains_default(self):
        self.controller.publish_remote_checkpoint.return_value={'id':'checkpoint'}
        with patch('subnet.persistent_training_protocol.independently_commit',return_value={'optimizer_steps':4})as commit:
            _,pointer,timing=complete(self.controller,self.report,self.job,{},'/original')
        self.assertIsNone(timing);self.assertEqual(pointer['optimizer_steps'],4);commit.assert_called_once()
    def test_strict_policy_bounds(self):
        for change in ({'checkpoint_readback_workers':True},{'checkpoint_readback_workers':5},
                       {'state_readback':'trainer-self'},{'version':'unknown'}):
            with self.subTest(change=change),self.assertRaises(ValueError):validate_policy(dict(self.policy,**change))

class ReaderAdmissionControls(unittest.TestCase):
    def setUp(self):
        self.key=SigningKey.generate();self.authority=bytes(self.key.verify_key).hex()
        host=dict(provider_UUID='reader',ssh_host_key_sha256='a'*64,evidence_sha256='b'*64)
        hashes={n:'c'*64 for n in ('remote_optimizer_readback.py','helper.py','supervisor.py')}
        payload=dict(version=ADMISSION,reader_identity='d'*64,reader_host_record_sha256=r.sha(host),
            module_hashes=hashes,all_23_objects_full_hash=True,qualification_evidence_sha256='e'*64,qualified_at=100)
        self.config=dict(endpoint=dict(host='example',port=22,user='root',known_hosts='/reader',python='/venv/python',workspace='/workspace',namespace='/reader-code'),
            reader_host=host,trainer_host=dict(host,provider_UUID='trainer'),trainer_known_hosts='/trainer',
            reader_identity='d'*64,module_hashes=hashes,qualification=r.sign(payload,self.key),max_wall_seconds=3500)
        self.controller=SimpleNamespace(authority=SimpleNamespace(id=self.authority))
    def test_actual_qualification_root_signature_required(self):
        IndependentStateReader(self.config,self.controller)
        self.config['qualification']=r.sign(self.config['qualification']['payload'],SigningKey.generate())
        with self.assertRaises(ValueError):IndependentStateReader(self.config,self.controller)
    def test_integer_fullhash_claim_is_rejected(self):
        payload=dict(self.config['qualification']['payload'],all_23_objects_full_hash=1)
        self.config['qualification']=r.sign(payload,self.key)
        with self.assertRaises(ValueError):IndependentStateReader(self.config,self.controller)
    def test_shared_provider_machine_rejected(self):
        self.config['trainer_host']['provider_UUID']='reader'
        with self.assertRaises(ValueError):IndependentStateReader(self.config,self.controller)
    def test_structured_launch_compiles_with_ROOT_literal_in_path(self):
        code=launch_code(dict(run='/ROOT-SIGNED/namespace',python='/venv/python',supervisor='/code/supervisor.py',launch='/ROOT-SIGNED/launch.ROOT-SIGNED.private.json',authority='a'*64,request_sha256='b'*64,launch_sha256='c'*64))
        compile(code,'prospective-reader-launch','exec');self.assertIn('launch.ROOT-SIGNED.private.json',code)
    def test_structured_dispatch_fake_spawn_records_original_PID_ticks(self):
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory)/'supervision'
            data=dict(run=str(root),python='/venv/python',supervisor='/code/supervisor.py',launch='/ROOT-SIGNED/launch.ROOT-SIGNED.private.json',authority='a'*64,request_sha256='b'*64,launch_sha256='c'*64)
            code=launch_code(data)
            with patch('subprocess.Popen',return_value=SimpleNamespace(pid=os.getpid()))as spawn, \
                 patch('os.nice')as nice,patch('os.getpriority',return_value=19), \
                 patch.dict(os.environ,{},clear=False),contextlib.redirect_stdout(io.StringIO()):
                exec(compile(code,'prospective-fake-dispatch','exec'),{})
            nice.assert_called_once_with(19);spawn.assert_called_once()
            self.assertEqual(spawn.call_args.args[0][5],'/ROOT-SIGNED/launch.ROOT-SIGNED.private.json')
            marker=json.loads((root/'original-supervisor-launch.private.json').read_text())
            self.assertEqual(marker['pid'],os.getpid());self.assertTrue(marker['ticks'].isdigit())
            with patch('subprocess.Popen')as duplicate,self.assertRaises(FileExistsError):exec(compile(code,'duplicate-fake-dispatch','exec'),{})
            duplicate.assert_not_called()
    def test_private_dispatch_record_0600_even_with_permissive_parent_umask(self):
        with tempfile.TemporaryDirectory()as directory:
            path=Path(directory)/'request.private.json';previous=os.umask(0)
            try:write_once(path,{'scoped':'test capability'})
            finally:os.umask(previous)
            self.assertEqual(path.stat().st_mode&0o777,0o600)
            with self.assertRaises(FileExistsError):write_once(path,{'replacement':True})
    def test_missing_or_future_qualification_time_is_rejected(self):
        for value in (0,True,time.time()+1000):
            payload=dict(self.config['qualification']['payload'],qualified_at=value)
            self.config['qualification']=r.sign(payload,self.key)
            with self.subTest(value=value),self.assertRaises(ValueError):IndependentStateReader(self.config,self.controller)
