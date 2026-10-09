import base64
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import time
import unittest
from unittest.mock import patch, MagicMock
from nacl.signing import SigningKey
from subnet import source_bootstrap as b

URL='https://test.r2.cloudflarestorage.com/bucket/current?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=test'

def archive(entries=None):
    entries=entries or [('subnet/__init__.py',b''),('subnet/cli.py',b'pass\n'),(b.TASK_ASSET,b'[]')]
    data=io.BytesIO()
    with tarfile.open(fileobj=data,mode='w:gz') as tar:
        for item in entries:
            if isinstance(item,tarfile.TarInfo):tar.addfile(item);continue
            name,body=item; member=tarfile.TarInfo(name);member.size=len(body);tar.addfile(member,io.BytesIO(body))
    body=data.getvalue();return body,{'sha256':hashlib.sha256(body).hexdigest(),'size':len(body),'url':URL}

def envelope(value,key):
    return b.canonical(dict(payload=value,signer=key.verify_key.encode().hex(),signature=base64.b64encode(key.sign(b.canonical(value)).signature).decode()))

class BootstrapTests(unittest.TestCase):
    def test_signed_source_url_aliases_are_unambiguous_and_https_only(self):
        for descriptor in ({'url':URL}, {'read_url':URL}, {'url':URL,'read_url':URL}):
            self.assertEqual(b.source_download_url(descriptor),URL)
        for descriptor in ({}, {'url':None,'read_url':URL}, {'url':URL,'read_url':URL+'&other=1'},
                           {'read_url':'http://test.r2.cloudflarestorage.com/source'},
                           {'read_url':'https://example.com/source'}):
            with self.assertRaises(ValueError):b.source_download_url(descriptor)

    def test_read_url_only_opening_reaches_admitted_source_without_reading_key(self):
        body,descriptor=archive();descriptor['read_url']=descriptor.pop('url')
        with tempfile.TemporaryDirectory() as tmp:
            key=Path(tmp)/'fixture-key';key.write_text('opaque fixture')
            argv=['--authority','a'*64,'--current-url',URL,'--source-cache',str(Path(tmp)/'cache'),
                  '--key',str(key),'--state',str(Path(tmp)/'miner'),'--once']
            with patch.object(b,'manifest',return_value={'source_bundle':descriptor}),patch.object(b,'download',return_value=body) as fetch,patch.object(b,'execute') as execute:
                b.main(argv)
            fetch.assert_called_once_with(URL,b.COMPRESSED_LIMIT)
            execute.assert_called_once()
            b.verify_cache(execute.call_args.args[0],b.admitted_files(body,descriptor))

    def test_explicit_search_preferences_reach_admitted_cli(self):
        body,descriptor=archive()
        with tempfile.TemporaryDirectory() as tmp:
            key=Path(tmp)/'fixture-key';key.write_text('opaque fixture, never read')
            argv=['--authority','a'*64,'--current-url',URL,'--source-cache',str(Path(tmp)/'cache'),
                '--key',str(key),'--state',str(Path(tmp)/'miner'),
                '--env-id','affine_math','--indices','0','2','--search-budget','7','--max-batches','1','--once']
            with patch.object(b,'manifest',return_value={'source_bundle':descriptor}),patch.object(b,'download',return_value=body),patch.object(b,'execute') as execute:
                b.main(argv)
            arguments=execute.call_args.args[1]
            self.assertEqual(arguments[arguments.index('--env-id')+1],'affine_math')
            self.assertEqual(arguments[arguments.index('--indices')+1:arguments.index('--search-budget')],['0','2'])
            self.assertEqual(arguments[arguments.index('--search-budget')+1],'7')
            self.assertIn('--source-bundle-sha256',arguments)
            b.verify_cache(execute.call_args.args[0],b.admitted_files(body,descriptor))

    def test_signatures_discovery_and_expiry(self):
        key=SigningKey.generate(); authority=key.verify_key.encode().hex()
        current=dict(epoch='math1',transport_policy='direct-r2-v1',manifest_url=URL)
        value=dict(epoch='math1',transport_policy='direct-r2-v1',deadline=time.time()+100,source_bundle={})
        responses=[envelope(current,key),envelope(value,key)]
        self.assertEqual(b.manifest(URL,authority,lambda *a:responses.pop(0)),value)
        for raw in [envelope(value,SigningKey.generate()),envelope(value,key)[:-4]]:
            with self.assertRaises(Exception):b.signed(raw,authority)
        for changed in [dict(value,epoch='wrong'),dict(value,transport_policy='gateway'),dict(value,deadline=0)]:
            responses=[envelope(current,key),envelope(changed,key)]
            with self.assertRaises(ValueError):b.manifest(URL,authority,lambda *a:responses.pop(0))

    def test_integrity_refuses_before_creating_cache(self):
        body,descriptor=archive()
        with tempfile.TemporaryDirectory() as tmp:
            cache=Path(tmp)/'uncreated'
            for bad,desc in [(body[:-2],descriptor),(body,dict(descriptor,sha256='0'*64)),(body,dict(descriptor,size=True)),(body,dict(descriptor,size=b.COMPRESSED_LIMIT+1))]:
                with self.assertRaises(ValueError):b.install(bad,desc,cache)
                self.assertFalse(cache.exists())

    def test_membership_traversal_duplicates_and_links(self):
        good=[('subnet/__init__.py',b''),('subnet/cli.py',b'pass'),(b.TASK_ASSET,b'[]')]
        for name in ['../escape','/absolute','subnet/../escape','subnet//x','subnet/.env','state/private.json','assets/other.json','ops/credentials/token','subnet/a.seed','subnet/a\\b']:
            body,desc=archive(good+[(name,b'private')])
            with self.assertRaises(ValueError):b.admitted_files(body,desc)
        for kind in [tarfile.SYMTYPE,tarfile.LNKTYPE,tarfile.DIRTYPE,tarfile.FIFOTYPE]:
            member=tarfile.TarInfo('subnet/link');member.type=kind;member.linkname='elsewhere'
            body,desc=archive(good+[member])
            with self.assertRaises(ValueError):b.admitted_files(body,desc)
        for additions in [[('subnet/cli.py',b'other')],[('./subnet/cli.py',b'other')],[('subnet/cli.py/x',b'other')]]:
            body,desc=archive(good+additions)
            with self.assertRaises(ValueError):b.admitted_files(body,desc)

    def test_expanded_stream_and_file_budgets(self):
        body,desc=archive()
        with patch.object(b,'RAW_LIMIT',64):
            with self.assertRaises(ValueError):b.admitted_files(body,desc)
        with patch.object(b,'MAX_FILES',2):
            with self.assertRaises(ValueError):b.admitted_files(body,desc)
        trailing=gzip.compress(gzip.decompress(body)+b'X'*1024)
        desc=dict(desc,size=len(trailing),sha256=hashlib.sha256(trailing).hexdigest())
        with patch.object(b,'RAW_LIMIT',1024):
            with self.assertRaises(ValueError):b.admitted_files(trailing,desc)

    def test_cache_full_bytes_membership_and_symlinks(self):
        body,desc=archive()
        with tempfile.TemporaryDirectory() as tmp:
            cache=Path(tmp)/'sources';source=b.install(body,desc,cache)
            self.assertEqual(source,b.install(body,desc,cache))
            self.assertEqual(source.name,desc['sha256'])
            source.chmod(0o700);file=source/'subnet/cli.py';file.chmod(0o600);file.write_bytes(b'tampered')
            with self.assertRaises(ValueError):b.install(body,desc,cache)
            file.write_bytes(b'pass\n');(source/'extra').write_text('unexpected')
            with self.assertRaises(ValueError):b.install(body,desc,cache)
            (source/'extra').unlink();file.parent.chmod(0o700);file.unlink();file.symlink_to('/etc/hosts')
            with self.assertRaises(ValueError):b.install(body,desc,cache)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'link';path.symlink_to(tmp)
            with self.assertRaises(ValueError):b.install(body,desc,path/'cache')

    def test_transport_bounds_and_redirects(self):
        for url in ['http://test.r2.cloudflarestorage.com/x','https://evil.test/x',URL+'&x=ok#fragment',URL.replace('https://','https://user@')]:
            with self.assertRaises(ValueError):b.r2_url(url)
        for status,headers,chunks in [(302,{},[]),(200,{'Content-Length':'4'},[b'ab']),(200,{},[b'abcde']),(200,{'Content-Encoding':'gzip'},[])]:
            response=MagicMock();response.__enter__.return_value=response;response.status_code=status;response.headers=headers;response.iter_content.return_value=chunks
            with patch.object(b.requests,'get',return_value=response) as get:
                with self.assertRaises(ValueError):b.download(URL,4)
                self.assertFalse(get.call_args.kwargs['allow_redirects'])
                self.assertEqual(get.call_args.kwargs['headers'],{'Accept-Encoding':'identity'})

    def test_identity_response_is_requested_and_exact_bytes_retained(self):
        response=MagicMock();response.__enter__.return_value=response
        response.status_code=200;response.headers={'Content-Length':'4'}
        response.iter_content.return_value=[b'ab',b'cd']
        with patch.object(b.requests,'get',return_value=response) as get:
            self.assertEqual(b.download(URL,4),b'abcd')
            self.assertEqual(get.call_args.kwargs['headers'],{'Accept-Encoding':'identity'})
            self.assertFalse(get.call_args.kwargs['allow_redirects'])

    def test_real_fresh_isolated_execution_of_approved_public_fixture(self):
        # Genuine subprocess qualification of admission/exec, deliberately no model/GPU.
        cli=b"import json,os,sys; print(json.dumps({'isolated':sys.flags.isolated,'bytecode':sys.dont_write_bytecode,'cwd':os.getcwd(),'args':sys.argv[1:],'pythonpath':os.environ.get('PYTHONPATH')}))\n"
        body,desc=archive([('subnet/__init__.py',b''),('subnet/cli.py',cli),(b.TASK_ASSET,b'[]')])
        with tempfile.TemporaryDirectory() as tmp:
            source=b.install(body,desc,Path(tmp)/'cache')
            script="from subnet.source_bootstrap import execute;execute("+repr(str(source))+",['--key','/explicit/key','--state','/explicit/state'])"
            env=dict(os.environ,PYTHONPATH=os.getcwd())
            result=subprocess.run([sys.executable,'-B','-c',script],env=env,capture_output=True,text=True,check=True)
            output=json.loads(result.stdout)
            self.assertEqual(output['isolated'],1);self.assertTrue(output['bytecode']);self.assertEqual(output['cwd'],str(source));self.assertIsNone(output['pythonpath'])
            self.assertEqual(output['args'],['--key','/explicit/key','--state','/explicit/state'])
            self.assertFalse(list(source.rglob('__pycache__')))

    def test_changed_source_refuses_before_model_or_checkpoint(self):
        import importlib.util
        from types import SimpleNamespace, ModuleType
        client=ModuleType('subnet.client');client.identity=MagicMock(return_value=SimpleNamespace(id='miner'));client.fetch_signed=MagicMock(return_value={'epoch':'new','source_bundle':{'sha256':'b'*64}});client.checkpoint_download=MagicMock();client.direct_r2_url=MagicMock()
        miner=ModuleType('subnet.miner');miner.Miner=MagicMock();miner.EpochClosed=type('EpochClosed',(RuntimeError,),{})
        batches=ModuleType('subnet.batches');batches.UploadBudgetExceeded=type('UploadBudgetExceeded',(Exception,),{})
        model=ModuleType('subnet.model');model.check_runtime_profile=MagicMock()
        protocol=ModuleType('subnet.protocol');protocol.entries=MagicMock();protocol.sample_key=MagicMock()
        with patch.dict(sys.modules,{'subnet.client':client,'subnet.miner':miner,'subnet.batches':batches,'subnet.model':model,'subnet.protocol':protocol}):
            spec=importlib.util.spec_from_file_location('subnet.bootstrap_test_cli',Path(b.__file__).with_name('cli.py'));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            with tempfile.TemporaryDirectory() as tmp:
                args=SimpleNamespace(cap_file=None,key='explicit',state=tmp,manifest_url=URL,current_url=None,authority='a'*64,source_bundle_sha256='a'*64)
                with self.assertRaisesRegex(ValueError,'source changed'):module.run(args)
            client.checkpoint_download.assert_not_called();miner.Miner.assert_not_called();model.check_runtime_profile.assert_not_called()

    def test_main_preserves_explicit_key_and_source_pinning(self):
        body,desc=archive()
        with tempfile.TemporaryDirectory() as tmp:
            key=Path(tmp)/'explicit-key';key.write_bytes(b'do-not-read-or-modify')
            with patch.object(b,'manifest',return_value={'source_bundle':desc}),patch.object(b,'download',return_value=body),patch.object(b,'execute') as execute:
                b.main(['--authority','a'*64,'--current-url',URL,'--key',str(key),'--state',str(Path(tmp)/'state'),'--source-cache',str(Path(tmp)/'cache'),'--once','--max-batches','2','--compression-level','1'])
            args=execute.call_args.args[1]
            self.assertIn('--once',args);self.assertEqual(args[args.index('--source-bundle-sha256')+1],desc['sha256'])
            self.assertEqual(args[args.index('--compression-level')+1],'1')
            self.assertEqual(key.read_bytes(),b'do-not-read-or-modify')
            self.assertFalse((Path(tmp)/'state').exists())

    def test_main_forwards_delegated_capability_without_a_private_key(self):
        body,desc=archive()
        with tempfile.TemporaryDirectory() as tmp:
            cap=Path(tmp)/'epoch-capability.json';cap.write_bytes(b'opaque-epoch-scoped-capability')
            with patch.object(b,'manifest',return_value={'source_bundle':desc}),patch.object(b,'download',return_value=body),patch.object(b,'execute') as execute:
                b.main(['--authority','a'*64,'--current-url',URL,'--cap-file',str(cap),'--state',str(Path(tmp)/'state'),'--source-cache',str(Path(tmp)/'cache'),'--once'])
            args=execute.call_args.args[1]
            self.assertNotIn('--key',args)
            self.assertEqual(args[args.index('--cap-file')+1],str(cap))
            self.assertEqual(args[args.index('--source-bundle-sha256')+1],desc['sha256'])
            self.assertEqual(cap.read_bytes(),b'opaque-epoch-scoped-capability')
            b.verify_cache(execute.call_args.args[0],b.admitted_files(body,desc))

    def test_delegated_capability_path_must_exist_and_not_be_a_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing=Path(tmp)/'missing';target=Path(tmp)/'cap';target.write_bytes(b'cap')
            link=Path(tmp)/'symlink';link.symlink_to(target)
            for cap in [missing,link]:
                with self.subTest(cap=cap.name),patch.object(b,'manifest') as fetch:
                    with self.assertRaisesRegex(ValueError,'credential file'):
                        b.main(['--authority','a'*64,'--current-url',URL,'--cap-file',str(cap),'--state',str(Path(tmp)/'state'),'--source-cache',str(Path(tmp)/'cache')])
                    fetch.assert_not_called()

if __name__=='__main__':unittest.main()
