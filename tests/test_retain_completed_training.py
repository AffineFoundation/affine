import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from ops.retain_completed_training import admitted_training,submission_archives,guard,sha
from ops.retain_verifier_downloads import verified_archive


class OriginalTrainingAdmission(unittest.TestCase):
    def setUp(self):
        self.digest='a'*64;self.miner='b'*64
        self.job=dict(role='train',job_id='original-train',steps=3,source_files={'subnet/model.py':'model-hash'},submissions=[{'sha256':self.digest}])
        self.manifest=dict(payable=False,epoch='epoch',source_bundle={'sha256':'source'},audit_frozen_receipts={self.miner:dict(sha256=self.digest,size=3,frozen_key='public/epoch/submissions/'+self.miner+'.zip')})
        self.sources={'source':dict(self.job['source_files'])}

    def test_only_bounded_original_training_with_approved_source_is_admitted(self):
        admitted_training(self.job,self.manifest,self.sources)
        for change in [{'role':'verify'},{'steps':0},{'steps':33},{'steps':True},{'job_id':'../train'},{'submissions':[]},{'submissions':[{}]*257},{'source_files':{'subnet/model.py':'substituted'}}]:
            with self.assertRaises(ValueError):admitted_training(dict(self.job,**change),self.manifest,self.sources)
        with self.assertRaises(ValueError):admitted_training(self.job,dict(self.manifest,payable=True),self.sources)
        with self.assertRaises(ValueError):admitted_training(self.job,self.manifest,{})

    def test_submissions_are_bound_to_exact_original_frozen_population(self):
        row=submission_archives(self.job,self.manifest)[0]
        self.assertEqual(row['index'],0);self.assertEqual(row['size'],3)
        self.assertEqual(row['archive_key'],'public/epoch/submissions/'+self.miner+'.zip')
        receipt=self.manifest['audit_frozen_receipts'][self.miner]
        for change in [{'size':0},{'size':2_000_000_001},{'frozen_key':'private/epoch/staging/'+self.miner+'.zip'},{'frozen_key':'public/other/submissions/'+self.miner+'.zip'}]:
            m=dict(self.manifest,audit_frozen_receipts={self.miner:dict(receipt,**change)})
            with self.assertRaises(ValueError):submission_archives(self.job,m)
        with self.assertRaises(ValueError):submission_archives(self.job,dict(self.manifest,audit_frozen_receipts={}))
        with self.assertRaises(ValueError):submission_archives(self.job,dict(self.manifest,audit_frozen_receipts={self.miner:receipt,'c'*64:receipt}))

    def test_full_archive_readback_rejects_truncation_or_corruption_and_closes_stream(self):
        import hashlib
        expected={'archive_key':'public/epoch/submissions/'+self.miner+'.zip','size':3,'sha256':hashlib.sha256(b'zip').hexdigest()}
        for content in [b'zi',b'bad',b'zipp']:
            body=io.BytesIO(content);client=SimpleNamespace(get_object=lambda **kw:{'Body':body,'ContentLength':3,'ETag':'etag'});bucket=SimpleNamespace(client=client,name='bucket')
            with self.assertRaises(ValueError):verified_archive(bucket,expected)
            self.assertTrue(body.closed)
        body=io.BytesIO(b'zip');bucket=SimpleNamespace(client=SimpleNamespace(get_object=lambda **kw:{'Body':body,'ContentLength':3,'ETag':'etag'}),name='bucket')
        self.assertTrue(verified_archive(bucket,expected)['archive_verified']);self.assertTrue(body.closed)

    def test_guard_requires_actual_process_ticks_and_config_bytes_and_protects_successor(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);state=root/'state';state.mkdir();config=root/'config.json';config.write_text(json.dumps({'state':str(state)}))
            (state/'controller.json').write_text(json.dumps({'checkpoint':{'id':'old'},'active':{'next_checkpoint':{'id':'new'}}}))
            record=root/'process.json';ticks=Path('/proc',str(os.getpid()),'stat').read_text().rsplit(')',1)[1].split()[19]
            r=dict(child_pid=os.getpid(),child_ticks=ticks,config_sha256=sha(config));record.write_text(json.dumps(r))
            self.assertEqual(guard(config,record)[1],{'old','new'})
            record.write_text(json.dumps(dict(r,child_ticks='wrong')))
            with self.assertRaises(ValueError):guard(config,record)
            record.write_text(json.dumps(r));config.write_text(config.read_text()+'\n')
            with self.assertRaises(ValueError):guard(config,record)


class CompleteOperatorCycle(unittest.TestCase):
    """Real isolated Python/HTTP archive path, with no SSH or external storage."""
    def test_actual_archive_readback_and_retirement_preserve_final_model_and_evidence(self):
        self.exercise_cycle()

    def test_corrupt_archive_blocks_all_retirement_and_preserves_local_weights(self):
        self.exercise_cycle(corrupt_archive=True)

    def exercise_cycle(self,corrupt_archive=False):
        import base64,hashlib,shlex,shutil,subprocess,sys,threading
        from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
        from nacl.signing import SigningKey
        from ops.retain_completed_training import run_cycle
        from subnet.storage import canonical
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);state=root/'state';roles=state/'roles';roles.mkdir(parents=True);workspace=root/'trainer';jobs=workspace/'jobs'/'epoch-train-original';jobs.mkdir(parents=True)
            key=SigningKey.generate();authority=key.verify_key.encode().hex();(state/'authority.seed').write_text(key.encode().hex())
            def sign(payload):return dict(payload=payload,signer=authority,signature=base64.b64encode(key.sign(canonical(payload)).signature).decode())
            def digest(payload):return hashlib.sha256(canonical(payload)).hexdigest()
            miner='b'*64;zipkey='public/epoch/submissions/'+miner+'.zip';ziphash=hashlib.sha256(b'zip').hexdigest();storage={zipkey:b'zip'}
            manifest=dict(epoch='epoch',payable=False,checkpoint={'id':'a'*64},source_bundle={'sha256':'source'},audit_frozen_receipts={miner:dict(sha256=ziphash,size=3,frozen_key=zipkey)})
            job=dict(job_id='epoch-train-original',role='train',steps=3,manifest=sign(manifest),source_files={'subnet/model.py':'approved'},submissions=[{'sha256':ziphash}]);envelope=sign(job)
            exports={}
            for step in (1,2,3):
                p=jobs/('checkpoint-step-'+str(step));p.mkdir();(p/'config.json').write_bytes(b'{}');(p/'model.safetensors').write_bytes(('model-'+str(step)).encode());exports[step]=digest({f.name:sha(f) for f in p.iterdir()})
            (jobs/'submission-0.zip').write_bytes(b'zip');final=exports[3]
            report=dict(job_id=job['job_id'],role='train',success=True,operator=authority,job_sha256=digest(job),epoch='epoch',checkpoint='a'*64,new_checkpoint={'id':final})
            terminal=dict(job_id=job['job_id'],phase='complete',exit_code=0,runner_pid=999999998,child_pid=999999999,runner_pid_ticks='0',child_pid_ticks='0')
            (workspace/'runner-status').mkdir();(workspace/'runner-status'/(job['job_id']+'.json')).write_bytes(canonical(terminal));(workspace/(job['job_id']+'.json')).write_bytes(canonical(envelope));(jobs/'report.json').write_bytes(canonical(report))
            (roles/(job['job_id']+'-job.json')).write_bytes(canonical(envelope));(roles/(job['job_id']+'-report.json')).write_bytes(canonical(report));(roles/'epoch-train.json').write_bytes(canonical({'job_id':job['job_id']}))
            (state/'controller.json').write_bytes(canonical({'checkpoint':{'id':'a'*64},'active':{'next_checkpoint':{'id':final}}}))
            config=root/'config.json';config.write_bytes(canonical({'state':str(state),'remote':{'roles':{'train':dict(user='root',host='fixture',port=1,known_hosts='/fixture',python=sys.executable,workspace=str(workspace))}},'bucket':{}}));config.chmod(0o600)
            record=root/'process.json';record.write_bytes(canonical(dict(child_pid=os.getpid(),child_ticks=Path('/proc',str(os.getpid()),'stat').read_text().rsplit(')',1)[1].split()[19],config_sha256=sha(config))))
            writer=root/'writer.json';writer.write_bytes(canonical(sign({'compute_state':str(state)})))
            binpath=root/'bin';binpath.mkdir();nvidia=binpath/'nvidia-smi';nvidia.write_text('#!/bin/sh\nexit 0\n');nvidia.chmod(0o700)
            class Handler(BaseHTTPRequestHandler):
                def log_message(self,*args):pass
                def do_PUT(self):
                    name=self.path[1:];body=self.rfile.read(int(self.headers['Content-Length']))
                    if name in storage:self.send_response(412)
                    else:storage[name]=body;self.send_response(200)
                    self.end_headers()
            server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();self.addCleanup(server.server_close);self.addCleanup(server.shutdown)
            class FakeClient:
                def get_object(self,**kwargs):
                    data=storage[kwargs['Key']]
                    if corrupt_archive and kwargs['Key'].endswith('/model.safetensors'):data=b'fakebad'
                    return dict(Body=io.BytesIO(data),ContentLength=len(data),ETag='fixture')
                def put_object(self,**kwargs):
                    self_case.assertNotIn(kwargs['Key'],storage);self_case.assertEqual(kwargs['IfNoneMatch'],'*');storage[kwargs['Key']]=kwargs['Body'];return {}
            class FakeBucket:
                name='fixture';client=FakeClient()
                def presign(self,name,*args,**kwargs):return 'http://127.0.0.1:'+str(server.server_port)+'/'+name
                def get(self,name):return storage[name]
            self_case=self;original_run=subprocess.run
            def local_transport(argv,**kwargs):
                if argv[0]=='scp':
                    shutil.copyfile(argv[-2],argv[-1].split(':',1)[1]);return subprocess.CompletedProcess(argv,0,'','')
                self.assertEqual(argv[0],'ssh');command=shlex.split(argv[-1]);self.assertEqual(command[:4],[sys.executable,'-I','-B','-c'])
                code=command[4]
                # Inspect actual descriptors/maps in this subprocess; avoid
                # reading other users' processes on shared test hosts.
                if 'spec.loader.exec_module(m)' in code:code=code.replace('spec.loader.exec_module(m)','spec.loader.exec_module(m);m.processes=lambda:[Path("/proc",str(os.getpid()))]')
                env=dict(os.environ,PATH=str(binpath)+os.pathsep+os.environ['PATH']);return original_run(command[:4]+[code],env=env,**kwargs)
            with patch('ops.retain_completed_training.Bucket',return_value=FakeBucket()),patch('ops.retain_completed_training.approved_source_members',return_value={'source':job['source_files']}),patch('ops.retain_completed_training.RemoteJobs.checked'),patch('ops.retain_completed_training.subprocess.run',side_effect=local_transport):
                if corrupt_archive:
                    with self.assertRaisesRegex(ValueError,'archive bytes changed'):run_cycle(config,writer,authority,root/'retention',record)
                    self.assertTrue(all((jobs/('checkpoint-step-'+str(i))/'model.safetensors').is_file() for i in (1,2,3)))
                    self.assertTrue((jobs/'submission-0.zip').is_file())
                    self.assertFalse(any(k.endswith('/descriptor.json') for k in storage))
                    return
                result=run_cycle(config,writer,authority,root/'retention',record)
            self.assertEqual(result['removed_replicas'],3);self.assertEqual(result['removed_bytes'],21)
            self.assertFalse((jobs/'checkpoint-step-1').exists());self.assertFalse((jobs/'checkpoint-step-2').exists());self.assertFalse((jobs/'submission-0.zip').exists())
            self.assertTrue((jobs/'checkpoint-step-3/model.safetensors').is_file());self.assertTrue((jobs/'report.json').is_file());self.assertTrue((workspace/(job['job_id']+'.json')).is_file())
            self.assertEqual(sum(k.endswith('/descriptor.json') for k in storage),2)
            self.assertEqual(storage[zipkey],b'zip')


if __name__=='__main__':unittest.main()
