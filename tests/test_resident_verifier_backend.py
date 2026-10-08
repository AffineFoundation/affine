import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('resident_operator',Path(__file__).resolve().parents[1]/'ops/resident_verifier_backend.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def job(cp='a',revision='fp32'):
    return dict(role='verify',source_files={'subnet/model.py':'s'},runtime_versions={'torch':'v'},
        manifest=dict(payload=dict(checkpoint=dict(id=cp,files={'weights':'digest'}),
            model_runtime_revision=revision,source_bundle=dict(sha256='source'),backend_profile={'sm':[9,0]})))

class ResidentTests(unittest.TestCase):
    def test_key_separates_checkpoint_source_runtime_and_profile(self):
        j=job();key=m.binding(j,'/tmp/science')
        for change in [lambda x:x['manifest']['payload']['checkpoint'].update(id='b'),
                       lambda x:x['manifest']['payload'].update(model_runtime_revision='bf16'),
                       lambda x:x['runtime_versions'].update(torch='other'),
                       lambda x:x['source_files'].update({'subnet/model.py':'changed'}),
                       lambda x:x['manifest']['payload'].update(backend_profile={'sm':[8,0]}),
                       lambda x:x['manifest']['payload']['source_bundle'].update(sha256='other')]:
            x=json.loads(json.dumps(j));change(x);self.assertNotEqual(key,m.binding(x,'/tmp/science'))
        self.assertNotEqual(key,m.binding(j,'/tmp/other'))
        j['role']='train'
        with self.assertRaises(ValueError):m.binding(j,'/tmp/science')

    def test_authenticate_once_but_refuse_changed_or_substituted_checkpoint(self):
        with tempfile.TemporaryDirectory()as t:
            root=Path(t);(root/'weights').write_bytes(b'valid')
            calls=[]
            backend=types.SimpleNamespace(checkpoint=lambda *a:(calls.append(1)or root),install_source_loader=lambda *a:None)
            r=m.ResidentRuntime(backend,root,m.binding(job(),root))
            manifest=job()['manifest']['payload']
            self.assertEqual(r.checkpoint(manifest,t),root)
            self.assertEqual(r.checkpoint(manifest,t),root)
            self.assertEqual(len(calls),1)
            (root/'weights').write_bytes(b'changed')
            with self.assertRaises(ValueError):r.checkpoint(manifest,t)

    def test_inventory_and_symlink_changes_fail_closed(self):
        with tempfile.TemporaryDirectory()as t:
            p=Path(t);(p/'weights').write_bytes(b'a');m.fingerprint(p,{'weights':'s'})
            (p/'extra').write_bytes(b'b')
            with self.assertRaises(ValueError):m.fingerprint(p,{'weights':'s'})
            (p/'extra').unlink();(p/'weights').unlink();(p/'weights').symlink_to('/etc/hosts')
            with self.assertRaises(ValueError):m.fingerprint(p,{'weights':'s'})

    def test_model_loaded_once_and_each_job_gets_clean_runtime(self):
        with tempfile.TemporaryDirectory()as t:
            p=Path(t);(p/'weights').write_bytes(b'w')
            calls=[]
            class FakeGPU:
                def __init__(self,*args,**kwargs):calls.append(kwargs['runtime_revision'])
                def for_environment(self,e,h):return types.SimpleNamespace(environment=e,harness=h)
            backend=types.SimpleNamespace(checkpoint=lambda *a:p,install_source_loader=lambda *a:None)
            r=m.ResidentRuntime(backend,p,m.binding(job(),p));r.checkpoint(job()['manifest']['payload'],t)
            with patch.dict('sys.modules',{'subnet.gpu_runtime':types.SimpleNamespace(GPURuntime=FakeGPU)}):
                a=r.factory(p,{'weights':'digest'},{'id':'one'},{'max':1})
                a.sampling_context='old miner'
                b=r.factory(p,{'weights':'digest'},{'id':'two'},{'max':2})
            self.assertEqual(calls,['fp32']);self.assertFalse(hasattr(b,'sampling_context'));self.assertEqual(b.environment,{'id':'two'})

    def test_lease_reused_and_closed_on_checkpoint_switch(self):
        from contextlib import contextmanager
        calls=[]
        class Lifecycle:
            root='/tmp/cache'
            @contextmanager
            def lease_checkpoint(self,cp):
                calls.append(('acquire',cp))
                try:yield 55
                finally:calls.append(('release',cp))
        client=m.ResidentClient();client.prepare(job(),'/tmp/source')
        for _ in range(2):
            with client.lease_checkpoint(Lifecycle(),'a')as fd:self.assertEqual(fd,55)
        self.assertEqual(calls,[('acquire','a')])
        client.prepare(job('b'),'/tmp/source');self.assertEqual(calls,[('acquire','a'),('release','a')])
        client.close()

    def test_idle_retirement_and_bounded_configuration(self):
        with self.assertRaises(ValueError):m.ResidentClient(idle_seconds=True)
        with self.assertRaises(ValueError):m.ResidentClient(idle_seconds=3601)
        c=m.ResidentClient(idle_seconds=1);c.prepare(job(),'/tmp/source');c.last_used-=2;c.idle();self.assertIsNone(c.key)

    def test_import_substitution_rejected_after_first_job(self):
        backend=types.SimpleNamespace(checkpoint=lambda *a:None,install_source_loader=lambda *a:None)
        r=m.ResidentRuntime(backend,'/tmp/frozen',m.binding(job(),'/tmp/frozen'));r.install_loader('/tmp/frozen')
        with patch.dict('sys.modules',{'subnet.model':types.SimpleNamespace(__file__='/tmp/attacker/subnet/model.py')}):
            with self.assertRaises(ValueError):r.install_loader('/tmp/frozen')


class ResidentProcessTests(unittest.TestCase):
    def test_two_jobs_use_one_original_process_and_one_model_load(self):
        import os,sys,subprocess
        with tempfile.TemporaryDirectory()as t:
            root=Path(t);source=root/'source';(source/'subnet').mkdir(parents=True)
            (source/'subnet/__init__.py').write_text('')
            cp=root/'checkpoint';cp.mkdir();(cp/'weights').write_bytes(b'w')
            (source/'subnet/backend_jobs.py').write_text('''import json,os
from pathlib import Path
signed=lambda e,a:e['payload']
load_job_envelope=lambda p,a:json.loads(Path(p).read_text())
def install_source_loader(*a):pass
def checkpoint(m,w,c=None):return Path(c)
def execute(envelope,authority,workspace,cache=None,runtime_factory=None):
 j=signed(envelope,authority)
 if j.get('crash'):os._exit(7)
 install_source_loader(Path(__file__).resolve().parents[1])
 p=checkpoint(j['manifest']['payload'],workspace,cache)
 runtime_factory(p,j['manifest']['payload']['checkpoint']['files'],{}, {})
 out=Path(workspace)/'jobs'/j['job_id'];out.mkdir(parents=True)
 (out/'report.json').write_text(json.dumps({'pid':os.getpid(),'job':j['job_id']}))
 return {'success':True}
''')
            (source/'subnet/gpu_runtime.py').write_text('''from pathlib import Path
class GPURuntime:
 def __init__(self,p,f,e,h,*,runtime_revision):
  counter=Path(p).parent/'loads';counter.write_text(str(int(counter.read_text())+1)if counter.exists()else'1')
 def for_environment(self,e,h):return self
''')
            c=m.ResidentClient();self.addCleanup(c.close);w=root/'work';w.mkdir();j=job();c.prepare(j,source);pids=[]
            for i in range(2):
                j['job_id']='job'+str(i);p=root/('job'+str(i)+'.json');p.write_text(json.dumps({'payload':j}))
                with (root/('log'+str(i))).open('xb')as log:
                    r=c.run([sys.executable,'-B','-m','subnet.backend_jobs',str(p),'--authority','test','--workspace',str(w),'--checkpoint-cache',str(cp)],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ),pass_fds=(),cwd=source)
                self.assertEqual(r.returncode,0);pids.append(json.loads((w/'jobs'/j['job_id']/'report.json').read_text())['pid'])
            self.assertEqual(pids[0],pids[1]);self.assertEqual((root/'loads').read_text(),'1')
            witnesses=[json.loads((w/'jobs'/('job'+str(i))/'resident-lifecycle.json').read_text())for i in range(2)]
            self.assertEqual([v['model_loads']for v in witnesses],[1,1])
            self.assertEqual([v['checkpoint_authentications']for v in witnesses],[1,1])
            self.assertEqual(witnesses[0]['model_object_id'],witnesses[1]['model_object_id'])
            # A child dying mid-original-attempt must not transparently replay.
            j['job_id']='uncertain';j['crash']=True;p=root/'crash.json';p.write_text(json.dumps({'payload':j}))
            with (root/'crash.log').open('xb')as log:
                with self.assertRaisesRegex(RuntimeError,'do not replay'):
                    c.run([sys.executable,'-B','-m','subnet.backend_jobs',str(p),'--authority','test','--workspace',str(w),'--checkpoint-cache',str(cp)],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ),pass_fds=(),cwd=source)
            self.assertEqual((root/'loads').read_text(),'1');c.close()

if __name__=='__main__':unittest.main()
