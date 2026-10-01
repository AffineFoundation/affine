import json,os,py_compile,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from subnet.environment_resources import collect_dependencies,export_dependencies,materialize,collect_task_resources
from subnet.adapter_resources import build_contract,validate_contract,verify_docker_before_start,prepare_task,prepare_environment

def reference(descriptor,sha='a'*64):
    return {'descriptor_id':descriptor['id'],'archive_sha256':sha,'object_key':'resources/'+descriptor['id']+'.zip'}

class AdapterResourceTests(unittest.TestCase):
    def test_controlled_scope_cannot_claim_full_closure(self):
        ref=reference({'id':'a'*64})
        with self.assertRaises(ValueError):build_contract(environment_version='prime-resource-full-v2',dependencies=ref,dependency_closure_reviewed=True)
        with self.assertRaises(ValueError):build_contract(environment_version='prime-resource-full-v2',dependencies=ref,dependency_scope='provider-namespace-controlled',dependency_closure_reviewed=False)
        with self.assertRaises(ValueError):build_contract(environment_version='prime-resource-controlled-v2',dependencies=ref,dependency_scope='provider-namespace-controlled',dependency_closure_reviewed=True)
        full=build_contract(environment_version='prime-resource-full-v2',dependencies=ref,dependency_closure_reviewed=True,native_platform_descriptor_id='b'*64)
        with self.assertRaises(ValueError):prepare_environment(full,{}, {},'/not-written',audience='verifier')
    def test_signed_docker_content_and_repository_digest(self):
        dep={'id':'a'*64}; image={'image_id':'sha256:'+'b'*64,'repo_digest':'example/image@sha256:'+'c'*64}
        contract=build_contract(environment_version='prime-resource-controlled-v2',dependencies=reference(dep),docker_image=image,dependency_scope='provider-namespace-controlled',dependency_closure_reviewed=False)
        self.assertEqual(verify_docker_before_start(contract,inspect=lambda _: {'Id':image['image_id'],'RepoDigests':[image['repo_digest']]}),image['image_id'])
        for actual in ({'Id':'sha256:'+'d'*64,'RepoDigests':[image['repo_digest']]},{'Id':image['image_id'],'RepoDigests':[]}):
            with self.assertRaises(ValueError):verify_docker_before_start(contract,inspect=lambda _,v=actual:v)
        forged=json.loads(json.dumps(contract));forged['docker_image']['image_id']='sha256:'+'d'*64
        with self.assertRaises(ValueError):validate_contract(forged)
        with self.assertRaises(ValueError):build_contract(environment_version='x',dependencies=reference(dep),docker_image={'image_id':'python:latest','repo_digest':None},dependency_scope='provider-namespace-controlled',dependency_closure_reviewed=False)

    def test_private_grader_and_portable_canonical_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);a=root/'host-a';b=root/'host-b'
            for p in (a,b):p.mkdir();(p/'grader.py').write_text('assert True\n')
            desc=collect_task_resources(a,origin={'git_commit':'123','path':'original/task'},audience='verifier')
            contract=build_contract(environment_version='prime-resource-controlled-v2',dependencies=reference({'id':'a'*64}),task_resources=[{'audience':'verifier','ref':reference(desc),'identity_resource':True}],dependency_scope='provider-namespace-controlled',dependency_closure_reviewed=False)
            descriptors={desc['id']:desc}; data={'idx':215,'task_dir':'/old/host/grader','question':'original question'}
            first=prepare_task(contract,descriptors,{desc['id']:str(a)},data,{},audience='verifier')
            second=prepare_task(contract,descriptors,{desc['id']:str(b)},{**data,'task_dir':'/another/host'}, {},audience='verifier')
            self.assertEqual(first['task_hash'],second['task_hash']);self.assertNotEqual(first['runtime_data']['task_dir'],second['runtime_data']['task_dir'])
            miner=prepare_task(contract,descriptors,{},data,{},audience='miner')
            self.assertEqual(miner['task_hash'],first['task_hash']);self.assertIsNone(miner['runtime_data'])
            (b/'grader.py').write_text('assert False\n')
            with self.assertRaises(ValueError):prepare_task(contract,descriptors,{desc['id']:str(b)},data,{},audience='verifier')

    def test_forged_bytecode_and_preloaded_provider_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);site=root/'site';site.mkdir();package=site/'probeprovider';package.mkdir()
            source=package/'__init__.py';source.write_text('VALUE=11\n')
            info=site/'probeprovider-1.dist-info';info.mkdir();(info/'METADATA').write_text('Name: probeprovider\nVersion: 1\n');(info/'RECORD').write_text('probeprovider/__init__.py,,\n')
            with patch.object(sys,'path',[str(site),*sys.path]):
                desc=collect_dependencies(['probeprovider']);archive=root/'deps.zip';receipt=export_dependencies(desc,archive)
            isolated=materialize(desc,archive,root/'isolated',archive_sha256=receipt['archive_sha256'])
            contract=build_contract(environment_version='prime-resource-controlled-v2',dependencies=reference(desc,receipt['archive_sha256']),dependency_scope='provider-namespace-controlled',dependency_closure_reviewed=False)
            manifest=root/'manifest.json';manifest.write_text(json.dumps({'contract':contract,'descriptors':{desc['id']:desc}}))
            env={**os.environ,'PYTHONPATH':str(isolated)+os.pathsep+str(Path(__file__).resolve().parents[1])}
            code="import json; from subnet.adapter_resources import verify_provider_before_import; d=json.load(open("+repr(str(manifest))+"));verify_provider_before_import(d['contract'],d['descriptors'],"+repr(str(isolated))+")"
            okay=subprocess.run([sys.executable,'-B','-c',code],env=env,capture_output=True,text=True);self.assertEqual(okay.returncode,0,okay.stderr)
            cached=isolated/'probeprovider'/'__pycache__';cached.mkdir();evil=root/'evil.py';evil.write_text('VALUE=99\n')
            stat=(isolated/'probeprovider/__init__.py').stat();os.utime(evil,(stat.st_atime,stat.st_mtime))
            pyc=Path(__import__('importlib.util',fromlist=['cache_from_source']).cache_from_source(str(isolated/'probeprovider/__init__.py')))
            py_compile.compile(str(evil),cfile=str(pyc),doraise=True)
            exploit=subprocess.run([sys.executable,'-B','-c','import probeprovider;print(probeprovider.VALUE)'],env=env,capture_output=True,text=True)
            self.assertEqual(exploit.stdout.strip(),'99')
            guarded=subprocess.run([sys.executable,'-B','-c',code],env=env,capture_output=True,text=True)
            self.assertNotEqual(guarded.returncode,0);self.assertIn('footprint mismatch',guarded.stderr)
            pyc.unlink();cached.rmdir()
            preloaded=subprocess.run([sys.executable,'-B','-c','import probeprovider;'+code],env=env,capture_output=True,text=True)
            self.assertNotEqual(preloaded.returncode,0);self.assertIn('already imported',preloaded.stderr)

    def test_miner_does_not_fetch_private_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);dep=collect_dependencies(['packaging']);archive=root/'deps.zip';receipt=export_dependencies(dep,archive)
            private=root/'private';private.mkdir();(private/'secret-grader.py').write_text('SECRET = 42\n')
            desc=collect_task_resources(private,origin={'git_commit':'original'},audience='verifier')
            contract=build_contract(environment_version='prime-resource-controlled-v2',dependencies=reference(dep,receipt['archive_sha256']),task_resources=[{'audience':'verifier','ref':reference(desc),'identity_resource':True}],dependency_scope='provider-namespace-controlled',dependency_closure_reviewed=False)
            prepared=prepare_environment(contract,{dep['id']:dep,desc['id']:desc},{dep['id']:archive},root/'cache',audience='miner')
            self.assertNotIn(desc['id'],prepared['resource_roots'])
            with self.assertRaises(ValueError):prepare_environment(contract,{dep['id']:dep,desc['id']:desc},{dep['id']:archive},root/'verifier-cache',audience='verifier')

if __name__=='__main__':unittest.main()
