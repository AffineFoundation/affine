import hashlib
import io
import json
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ops.continuous_owned_heldout128_ssh import Adapter,emit,STAGE,PROBE


class AdapterTests(unittest.TestCase):
    def test_full_emitted_scripts_compile_not_string_substitution(self):
        for body in (STAGE,PROBE):
            compile(emit(body,dict(root="SOURCE');raise RuntimeError('injected')#",runtime={})), 'script','exec')
    def test_actual_CPU_stager_fills_missing_members_preserves_equal_and_rejects_changed(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d)/'owned';root.mkdir();raw=b'class S:\n def reset(self,*a):pass\n def close(self):pass\ndef create_session(spec):return S()\n'
            files={'subnet/__init__.py':b'','subnet/environments.py':raw,'public-intended.txt':b'original'}
            archive=root/'source.tar.gz'
            with tarfile.open(archive,'w:gz')as t:
                for name,value in files.items():
                    m=tarfile.TarInfo(name);m.size=len(value);t.addfile(m,io.BytesIO(value))
            source=root/'source';source.mkdir();(source/'public-intended.txt').write_bytes(b'original')
            plan=dict(root=str(root),source_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                inventory={n:hashlib.sha256(v).hexdigest()for n,v in files.items()},cpu_files={},environment={},index=1)
            def run():return subprocess.run([sys.executable,'-I','-B','-'],input=emit(STAGE,plan),capture_output=True,text=True,timeout=15)
            result=run();self.assertEqual(result.returncode,0,result.stderr)
            self.assertTrue(json.loads(result.stdout)['native_CPU_ready']);self.assertFalse(json.loads(result.stdout)['torch_imported'])
            self.assertEqual(run().returncode,0)
            (source/'public-intended.txt').write_bytes(b'changed')
            self.assertNotEqual(run().returncode,0);self.assertEqual((source/'public-intended.txt').read_bytes(),b'changed')
    def test_actual_CPU_stage_does_not_follow_owned_member_symlink(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d)/'owned';root.mkdir();source=root/'source';source.mkdir();foreign=Path(d)/'foreign';foreign.write_bytes(b'foreign');(source/'member').symlink_to(foreign)
            archive=root/'source.tar.gz'
            with tarfile.open(archive,'w:gz')as t:
                m=tarfile.TarInfo('member');m.size=5;t.addfile(m,io.BytesIO(b'known'))
            plan=dict(root=str(root),source_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),inventory={'member':hashlib.sha256(b'known').hexdigest()},cpu_files={},environment={},index=1)
            r=subprocess.run([sys.executable,'-I','-B','-'],input=emit(STAGE,plan),capture_output=True,text=True,timeout=15)
            self.assertNotEqual(r.returncode,0);self.assertEqual(foreign.read_bytes(),b'foreign')
    def test_idle_rejects_changed_physical_GPU_or_runtime_and_defers_capacity(self):
        adapter=Adapter.__new__(Adapter);adapter.p=dict(runtime_versions={'torch':'approved'},machine_id_sha256='machine',gpu_uuid='gpu',minimum_free_cold_bytes=20,minimum_available_ram_bytes=64)
        good=dict(machine='machine',gpu='gpu',runtime={'torch':'approved'},active=[],compute='',free=30,ram=100)
        adapter.remote=lambda *a,**k:good
        self.assertTrue(adapter.idle())
        for key,value in [('machine','wrong'),('gpu','wrong'),('runtime',{'torch':'wrong'})]:
            old=good[key];good[key]=value
            with self.assertRaises(ValueError):adapter.idle()
            good[key]=old
        for key,value in [('active',[12]),('compute','12'),('free',19),('ram',63)]:
            old=good[key];good[key]=value;self.assertFalse(adapter.idle());good[key]=old
    def test_original_publication_hash_and_signature_rejection(self):
        import test_owned_cached_group_operator as fixture
        f=fixture.OperatorTests();f.setUp();self.addCleanup(f.doCleanups)
        adapter=Adapter.__new__(Adapter);adapter.authority=f.authority
        production=f.root/'production';production.mkdir();adapter.p=dict(production_directory=str(production))
        state=dict(version='authority-persistent-trainer-state-v1',descriptor={'optimizer_steps':14})
        metrics=dict(trainer_state=dict(descriptor_key='private-original-state',publication_sha256=f.digest(state)))
        path=production/'epoch-training-metrics.json';path.write_text(json.dumps(metrics));path.chmod(0o600)
        objects={'private-original-state':json.dumps(f.sign(state)).encode()}
        model_key='public/checkpoints/'+f.cp['id']+'/authorities/'+f.authority+'/checkpoint.json'
        objects[model_key]=json.dumps(f.sign(f.cp)).encode()
        class B:
            def get(self,key):return objects[key]
        adapter.bucket=B();row=dict(checkpoint=f.cp['id'],completion=f.sign(dict(epoch='epoch')))
        self.assertEqual(adapter.publication(row)['checkpoint_descriptor']['payload'],f.cp)
        objects['private-original-state']=json.dumps(f.sign(dict(state,extra='changed'))).encode()
        with self.assertRaises(ValueError):adapter.publication(row)
        changed=f.sign(state);changed['payload']['descriptor']['optimizer_steps']=15
        objects['private-original-state']=json.dumps(changed).encode()
        with self.assertRaises(Exception):adapter.publication(row)


if __name__=='__main__':unittest.main()
