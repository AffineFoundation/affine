import base64,copy,hashlib,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
from subnet.backend_jobs import canonical
from subnet.storage import Identity
from nacl.exceptions import BadSignatureError
from subnet.checkpoint_evaluator import QualifiedEvaluationJobs,SOURCE_ROUTES_VERSION,RemoteObservationTimeout,qualified_dispatcher,detached_historical_run

def seal(payload,identity):
    return dict(payload=payload,signer=identity.id,signature=base64.b64encode(identity.key.sign(canonical(payload)).signature).decode())

class RouteTimeout(TimeoutError):
    def __init__(self,job_id,role):self.job_id=job_id;self.role=role

class SourceRouting(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.state=self.root/'state';self.state.mkdir();(self.state/'roles').mkdir()
        self.authority=Identity(bytes(range(32)))
        self.controller=SimpleNamespace(state=self.state,authority=self.authority)
        self.rows={};self.remotes={};self.calls=[]
        for name,sha in [('old','a'*64),('new','b'*64),('pending','c'*64)]:
            root=self.root/name;(root/'subnet').mkdir(parents=True)
            for filename in ['__init__.py','backend_jobs.py','remote_backend.py']:(root/'subnet'/filename).write_text('# '+name+' '+filename+'\n')
            files={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()for p in(root/'subnet').glob('*.py')}
            row=dict(endpoint=dict(host='same-host',port=22,user='root',workspace='/original/evaluator',code='/'+name),local_source_path=str(root),source_files=files,runtime_versions=dict(torch='1',transformers='2',toploc='3'),new_dispatch_approved=name!='pending')
            self.rows[sha]=row
            self.remotes[sha]=SimpleNamespace(metadata={k:copy.deepcopy(row[k])for k in ['source_files','runtime_versions']},run=Mock(return_value={'original':name}),remote_status=Mock(return_value={'phase':'complete'}),observation_timeout_type=RouteTimeout)
        self.document=lambda:seal(dict(version=SOURCE_ROUTES_VERSION,physical_id='provider-original-uuid',sources=copy.deepcopy(self.rows)),self.authority)
        def factory(row,controller):
            sha=next(k for k,v in self.rows.items()if v['local_source_path']==row['local_source_path']);self.calls.append(sha);return self.remotes[sha]
        self.factory=factory
    def router(self):return QualifiedEvaluationJobs(self.controller,self.document(),self.factory)
    def manifest(self,sha):return dict(epoch='original-epoch',checkpoint={'id':'actual-parent'},source_bundle={'sha256':sha})
    def existing(self,sha,label='original'):
        m=self.manifest(sha);row=self.rows[sha];job=dict(job_id='original-job',role='evaluate',manifest=seal(m,self.authority),source_files=row['source_files'],runtime_versions=row['runtime_versions'],heldout=[{'seeds':[100]}])
        p=self.state/'roles';(p/'original-job-job.json').write_bytes(canonical(seal(job,self.authority)))
        (p/(label+'.json')).write_bytes(canonical(dict(job_id=job['job_id'],role='evaluate',job_sha256=hashlib.sha256(canonical(job)).hexdigest())))
        return m
    def test_dispatch_eligibility_authenticates_issued_original(self):
        r=self.router();sha='c'*64;request={'manifest':self.manifest(sha),'label':'original'}
        self.assertFalse(r.dispatch_eligible(request));self.existing(sha);self.assertTrue(r.dispatch_eligible(request));self.assertEqual(self.calls,[])
        path=self.state/'roles/original-job-job.json';value=json.loads(path.read_text());value['payload']['role']='train';path.write_text(json.dumps(value))
        with self.assertRaises(BadSignatureError):r.dispatch_eligible(request)
    def test_signed_disk_admission_blocks_new_job_but_not_original_adoption(self):
        sha='b'*64;self.rows[sha]['endpoint']['evaluation_min_free_disk_bytes']=32
        remote=self.remotes[sha];remote.python='python3';remote.command=Mock(return_value='{"free":31}')
        r=self.router()
        with self.assertRaisesRegex(OSError,'disk admission'):r.run('original','evaluate',self.manifest(sha))
        remote.run.assert_not_called();self.existing(sha);r.run('original','evaluate',self.manifest(sha),heldout=[{'seeds':[100]}]);remote.run.assert_called_once()
        self.assertEqual(remote.command.call_count,1)
    def test_two_source_routes_preserve_exact_labels_seeds_and_manifest(self):
        r=self.router();plan=[dict(indices=[17,42],seeds=[1700,4200])]
        for sha in ['a'*64,'b'*64]:
            manifest=self.manifest(sha);original=copy.deepcopy(manifest)
            r.run('original-label','evaluate',manifest,'original-cache',heldout=plan)
            self.assertEqual(manifest,original)
            self.remotes[sha].run.assert_called_once_with('original-label','evaluate',original,'original-cache',heldout=plan)
        self.assertEqual(self.calls,['a'*64,'b'*64])
    def test_unknown_source_and_non_evaluation_never_construct_or_dispatch(self):
        r=self.router()
        for role,sha in [('evaluate','d'*64),('train','a'*64)]:
            with self.assertRaises(ValueError):r.run('original',role,self.manifest(sha))
        self.assertEqual(self.calls,[])
    def test_unapproved_source_adopts_only_same_signed_original_request(self):
        r=self.router();sha='c'*64
        with self.assertRaisesRegex(ValueError,'new GPU'):r.run('original','evaluate',self.manifest(sha))
        self.assertEqual(self.calls,[])
        m=self.existing(sha);r.run('original','evaluate',m,heldout=[{'seeds':[100]}])
        self.remotes[sha].run.assert_called_once()
        altered=dict(m,checkpoint={'id':'relabelled'})
        with self.assertRaisesRegex(ValueError,'route changed'):r.run('original','evaluate',altered)
        self.assertEqual(self.remotes[sha].run.call_count,1)
    def test_historical_source_inventory_change_rejected_before_remote_dispatch(self):
        m=self.existing('a'*64);r=self.router();self.rows['a'*64]['source_files']['subnet/backend_jobs.py']='f'*64
        # The immutable signed original request still contains its real old map.
        job=self.state/'roles/original-job-job.json';v=json.loads(job.read_bytes());v['payload']['source_files']['subnet/backend_jobs.py']='e'*64;job.write_bytes(canonical(v))
        with self.assertRaises(BadSignatureError):r.run('original','evaluate',m)
        self.assertEqual(self.calls,[])
    def test_real_transport_timeout_retains_original_job_identifier(self):
        r=self.router();self.remotes['a'*64].run.side_effect=RouteTimeout('same-original','evaluate')
        with self.assertRaises(RemoteObservationTimeout)as e:r.run('original','evaluate',self.manifest('a'*64))
        self.assertEqual(e.exception.job_id,'same-original');self.assertEqual(self.remotes['a'*64].run.call_count,1)
    def test_signed_map_local_and_actual_remote_bytes_must_match(self):
        row=self.rows['a'*64];(Path(row['local_source_path'])/'subnet/backend_jobs.py').write_text('tampered')
        with self.assertRaisesRegex(ValueError,'local source hash'):self.router()
        self.assertEqual(self.calls,[])
    def test_wrong_remote_runtime_prevents_any_gpu_dispatch(self):
        r=self.router();self.remotes['a'*64].metadata['runtime_versions']['torch']='other'
        with self.assertRaisesRegex(ValueError,'metadata mismatch'):r.run('original','evaluate',self.manifest('a'*64))
        self.remotes['a'*64].run.assert_not_called()
    def test_map_signature_and_second_physical_evaluator_rejected(self):
        doc=self.document();doc['payload']['physical_id']='changed'
        with self.assertRaises(BadSignatureError):QualifiedEvaluationJobs(self.controller,doc,self.factory)
        self.rows['b'*64]['endpoint']['workspace']='/different/ledger'
        with self.assertRaisesRegex(ValueError,'one physical'):self.router()
    def test_busy_unknown_historical_source_reserves_physical_gpu_without_dispatch(self):
        m=self.existing('a'*64);p=self.state/'roles/original-job-job.json';j=json.loads(p.read_bytes())['payload'];j['manifest']=seal(self.manifest('d'*64),self.authority);p.write_bytes(canonical(seal(j,self.authority)))
        rec=self.state/'roles/original.json';v=json.loads(rec.read_bytes());v['job_sha256']=hashlib.sha256(canonical(j)).hexdigest();rec.write_bytes(canonical(v))
        r=self.router();self.remotes['a'*64].remote_status.return_value={'phase':'running'}
        self.assertTrue(r.busy());self.remotes['a'*64].run.assert_not_called()
    def test_approved_map_does_not_modify_callers_signed_document(self):
        doc=self.document();before=canonical(doc);QualifiedEvaluationJobs(self.controller,doc,self.factory)
        self.assertEqual(canonical(doc),before)

    def test_unknown_physical_phase_blocks_and_symlinked_runtime_rejected(self):
        self.existing('a'*64);r=self.router()
        self.remotes['a'*64].remote_status.return_value={'phase':'unknown'}
        with self.assertRaisesRegex(ValueError,'liveness'):r.busy()
        root=Path(self.rows['a'*64]['local_source_path']);(root/'subnet').rename(root/'real')
        (root/'subnet').symlink_to(root/'real',target_is_directory=True)
        with self.assertRaisesRegex(ValueError,'source tree'):self.router()
    def test_real_isolated_cpu_packages_preserve_distinct_historical_abis(self):
        import subnet.remote_backend as active
        original=active.RemoteJobs
        for sha,row in list(self.rows.items())[:2]:
            root=Path(row['local_source_path']); marker=sha[0]
            (root/'subnet/backend_jobs.py').write_text('ABI='+repr(marker)+'\n')
            (root/'subnet/remote_backend.py').write_text("from .backend_jobs import ABI\nclass RemoteObservationTimeout(TimeoutError):pass\nclass RemoteJobs:\n    def __init__(self,endpoint,controller):self.abi=ABI\n    def run(self,identifier,remotejob,cache=None):\n        self.launch_runner(identifier,remotejob,cache)\n")
            row['source_files']={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()for p in(root/'subnet').glob('*.py')}
            with patch.object(active.RemoteJobs,'launch_runner',autospec=True)as launch:
                remote=qualified_dispatcher(row,self.controller)
                remote.run('same-id','same-path','same-cache')
                launch.assert_called_once_with(remote,'same-id','same-path','same-cache')
            self.assertEqual(remote.abi,marker)
            self.assertIs(active.RemoteJobs,original)
    def test_actual_historical_source_launch_block_only_is_replaced(self):
        import ast,importlib.util
        path=Path('/home/const/subnet120-rewrite/state/root-audits/decoupled-learner-eight-verifier-source-preparation-20261005-v5/source-candidate/subnet/remote_backend.py')
        if not path.exists():self.skipTest('retained production historical CPU source unavailable')
        # Parse the genuine retained source without importing its dependencies.
        tree=ast.parse(path.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='RemoteJobs')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='run')
        import inspect,textwrap
        source='\n'.join(path.read_text().splitlines()[method.lineno-1:method.end_lineno])
        module=SimpleNamespace(RemoteJobs=SimpleNamespace(run=object()),__file__=str(path),__dict__={})
        with patch('subnet.checkpoint_evaluator.inspect.getsource',return_value=source):
            result=detached_historical_run(module)
        changed=result.__code__
        self.assertIn('launch_runner',changed.co_names)
        self.assertNotIn('nohup',changed.co_consts)
        with patch('subnet.checkpoint_evaluator.inspect.getsource',return_value='def run(self):\n    self.command("nohup unknown")\n'):
            with self.assertRaisesRegex(ValueError,'unreviewed'):detached_historical_run(module)

    def test_signed_original_heldout_cannot_be_changed_during_adoption(self):
        m=self.existing('a'*64);r=self.router()
        with self.assertRaisesRegex(ValueError,'route changed'):r.run('original','evaluate',m,heldout=[{'seeds':[101]}])
        self.assertEqual(self.calls,[])
    def test_live_original_blocks_new_dispatch_without_replacing_it(self):
        self.existing('a'*64);r=self.router()
        self.remotes['a'*64].remote_status.return_value={'phase':'running'}
        with self.assertRaises(RemoteObservationTimeout):r.run('next','evaluate',self.manifest('b'*64))
        self.remotes['b'*64].run.assert_not_called()

    def test_real_factory_rechecks_bytes_before_any_import_or_ssh(self):
        row=self.rows['a'*64];(Path(row['local_source_path'])/'subnet/backend_jobs.py').write_text('raise RuntimeError("must not import")')
        with self.assertRaisesRegex(ValueError,'bytes changed'):qualified_dispatcher(row,self.controller)
