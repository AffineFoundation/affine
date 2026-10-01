import base64,json,os,py_compile,subprocess,sys,tempfile,unittest,zipfile,time
from pathlib import Path
from nacl.signing import SigningKey
from subnet.environments import build_spec as build_environment,create_session
from subnet.environment_resources import canonical,collect_dependencies,export_dependencies,digest
from subnet.adapter_resources import build_contract
from subnet.resource_session import build_spec,ResourceSessionProxy,read_protocol_line

class ResourceSessionTests(unittest.TestCase):
    def test_actual_signed_original_task_and_worker_tampering(self):
        repo=Path(__file__).resolve().parents[1]
        original=build_environment('affine_verbatim',{'taskset':{'num_samples':2,'target_length':8,'content_type':'codes'}},num_samples=2,max_turns=1).to_dict()
        public_session=create_session(build_environment('affine_verbatim',original['config'],num_samples=2,max_turns=1))
        try:prompt=public_session.reset(0,0)['messages'][-1]['content']
        finally:public_session.close()
        dep=collect_dependencies(['verifiers'])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);archive=root/'deps.zip';receipt=export_dependencies(dep,archive)
            ref={'descriptor_id':dep['id'],'archive_sha256':receipt['archive_sha256'],'object_key':'resources/'+dep['id']+'.zip'}
            # Controlled provider-footprint conformance; production requires a
            # reviewed wider dependency/native platform closure separately.
            contract=build_contract(environment_version='prime-resource-controlled-v1',dependencies=ref,dependency_scope='provider-namespace-controlled',dependency_closure_reviewed=False)
            bridge=build_spec(original,contract);signer=SigningKey.generate();authority=signer.verify_key.encode().hex()
            envelope={'payload':bridge,'signer':authority,'signature':base64.b64encode(signer.sign(canonical(bridge)).signature).decode()}
            request={'signed_spec':envelope,'descriptors':{dep['id']:dep},'archives':{dep['id']:str(archive)},'cache':str(root/'cache'),'audience':'verifier'}
            requestpath=root/'request.json';requestpath.write_text(json.dumps(request))
            bootstrap='import sys,runpy;sys.path.insert(0,'+repr(str(repo))+');runpy.run_module("subnet.resource_session",run_name="__main__")'
            command=[sys.executable,'-I','-B','-c',bootstrap,'--request',str(requestpath),'--authority',authority]
            visible=prompt.rsplit('<text>',1)[1].split('</text>',1)[0]
            payload='\n'.join(json.dumps(x) for x in [{'op':'reset','index':0,'seed':0},{'op':'step','action':{'text':'<answer>'+visible+'</answer>'}},{'op':'close'}])+'\n'
            actual=subprocess.run(command,input=payload,capture_output=True,text=True,timeout=60)
            self.assertEqual(actual.returncode,0,actual.stderr)
            responses=[json.loads(line) for line in actual.stdout.splitlines()]
            self.assertEqual(responses[-1]['result']['reward'],1.0);self.assertEqual(responses[0]['result']['environment_version'],'prime-resource-controlled-v1')
            outer={**original,'adapter':'resource_prime_v1_controlled','version':contract['environment_version'],'source_hash':bridge['source_hash'],
                   'config':{'signed_resource_spec':envelope,'authority':authority}}
            proxy=ResourceSessionProxy(outer,{k:request[k] for k in ('descriptors','archives','cache','audience')})
            try:
                first=proxy.reset(0,0);self.assertEqual(first['task_hash'],responses[0]['result']['task_hash'])
                self.assertEqual(proxy.step({'text':'<answer>'+visible+'</answer>'})['reward'],1.0)
            finally:proxy.close()
            # -I prevents hostile PYTHONPATH/sitecustomize from executing before
            # the authenticated provider guard; selected source still grades1.
            hostile=root/'hostile';hostile.mkdir();(hostile/'sitecustomize.py').write_text('raise RuntimeError("hostile bootstrap")\n')
            shadow=hostile/'verifiers';shadow.mkdir();(shadow/'__init__.py').write_text('raise RuntimeError("shadow provider")\n')
            withshadow=subprocess.run(command,input=payload,capture_output=True,text=True,timeout=60,env={**os.environ,'PYTHONPATH':str(hostile)})
            self.assertEqual(withshadow.returncode,0,withshadow.stderr)
            self.assertEqual(json.loads(withshadow.stdout.splitlines()[-1])['result']['reward'],1.0)
            # Authenticated controlled SDK variant emits native FD1 bytes while
            # importing the actual provider. These go to diagnostics, not IPC.
            with zipfile.ZipFile(archive) as z:entries={name:z.read(name) for name in z.namelist()}
            noisy=json.loads(json.dumps(dep));entry='files/verifiers/__init__.py';entries[entry]+=b'\nimport os; os.write(1, b"native-provider-noise-without-newline")\n'
            noisy['packages'][0]['files']['verifiers/__init__.py']=digest(entries[entry]);noisy['id']=digest(canonical({k:v for k,v in noisy.items() if k!='id'}));entries['DESCRIPTOR.json']=canonical(noisy)
            noisyarchive=root/'noise.zip'
            with zipfile.ZipFile(noisyarchive,'w') as z:
                for name,data in entries.items():z.writestr(name,data)
            noisyref={'descriptor_id':noisy['id'],'archive_sha256':digest(noisyarchive.read_bytes()),'object_key':'resources/noisy-controlled.zip'}
            noisycontract=build_contract(environment_version='prime-resource-controlled-v1',dependencies=noisyref,dependency_scope='provider-namespace-controlled',dependency_closure_reviewed=False)
            noisyspec=build_spec(original,noisycontract);noisyenvelope={'payload':noisyspec,'signer':authority,'signature':base64.b64encode(signer.sign(canonical(noisyspec)).signature).decode()}
            noisyrequest={'signed_spec':noisyenvelope,'descriptors':{noisy['id']:noisy},'archives':{noisy['id']:str(noisyarchive)},'cache':str(root/'noise-cache'),'audience':'verifier'}
            requestpath.write_text(json.dumps(noisyrequest));noisyrun=subprocess.run(command,input=payload,capture_output=True,text=True,timeout=60)
            self.assertEqual(noisyrun.returncode,0,noisyrun.stderr);self.assertIn('native-provider-noise',noisyrun.stderr)
            self.assertNotIn('native-provider-noise',noisyrun.stdout);self.assertEqual(json.loads(noisyrun.stdout.splitlines()[-1])['result']['reward'],1.0)
            forged=json.loads(json.dumps(request));forged['signed_spec']['payload']['execution_resources']['dependency_closure_reviewed']=True
            requestpath.write_text(json.dumps(forged));bad=subprocess.run(command,input=payload,capture_output=True,text=True,timeout=20)
            self.assertNotEqual(bad.returncode,0);self.assertIn('Signature',bad.stderr)
            requestpath.write_text(json.dumps(request))
            # Existing materialized cache receives a real timestamp/size-matched
            # hostile pyc: reject exact cache membership BEFORE task construction.
            provider=root/'cache'/dep['id']/'verifiers/__init__.py';evil=root/'evil.py';evil.write_bytes(b'X'*(provider.stat().st_size-1)+b'\n')
            # Compilable malicious source with identical source-file size.
            evil.write_text('VALUE=99\n'+' '*(provider.stat().st_size-len('VALUE=99\n')))
            st=provider.stat();os.utime(evil,(st.st_atime,st.st_mtime));pyc=Path(__import__('importlib.util',fromlist=['cache_from_source']).cache_from_source(str(provider)));pyc.parent.mkdir()
            py_compile.compile(str(evil),cfile=str(pyc),doraise=True)
            badcache=subprocess.run(command,input=payload,capture_output=True,text=True,timeout=20)
            self.assertNotEqual(badcache.returncode,0);self.assertIn('footprint mismatch',badcache.stderr)
            pyc.unlink();pyc.parent.rmdir()
            provider.parent.rename(root/'saved-verifiers')
            (root/'cache'/dep['id']/'verifiers').symlink_to(shadow,target_is_directory=True)
            badshadow=subprocess.run(command,input=payload,capture_output=True,text=True,timeout=20)
            self.assertNotEqual(badshadow.returncode,0);self.assertIn('symlink',badshadow.stderr)

    def test_partial_protocol_frame_has_real_deadline(self):
        read,write=os.pipe();stream=os.fdopen(read,'rb',buffering=0)
        try:
            os.write(write,b'{"ok":true');started=time.monotonic()
            with self.assertRaises(TimeoutError):read_protocol_line(stream,timeout=.05)
            self.assertLess(time.monotonic()-started,.5)
        finally:stream.close();os.close(write)

if __name__=='__main__':unittest.main()
