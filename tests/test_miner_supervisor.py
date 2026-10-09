import base64,copy,fcntl,hashlib,io,json,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
from nacl.signing import SigningKey
from subnet import miner_supervisor as s
from test_source_bootstrap import archive


class Response:
    def __init__(self,body,headers=None,status=206):
        self.raw=io.BytesIO(body);self.body=body;self.headers=headers or {};self.status_code=status
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def iter_content(self,n):yield self.body
    def close(self):pass


class Session:
    def __init__(self,files,fail_once=False):self.files=files;self.requests=[];self.fail_once=fail_once
    def get(self,url,headers=None,**kwargs):
        import re,requests
        name=url.split('?')[0].rsplit('/',1)[1];data=self.files[name]
        start,end=map(int,re.fullmatch(r'bytes=(\d+)-(\d+)',headers['Range']).groups())
        self.requests.append((name,start,end))
        result=Response(data[start:end+1],{'Content-Range':f'bytes {start}-{end}/{len(data)}','Content-Length':str(end-start+1)})
        if self.fail_once and end>2:
            self.fail_once=False
            def cut(n):
                yield data[start:start+2]
                raise requests.exceptions.ChunkedEncodingError('interrupted fixture')
            result.iter_content=cut
        return result
    def close(self):pass


class SupervisorTests(unittest.TestCase):
    def fixture(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.data={'config.json':b'{}','model.safetensors':b'0123456789abcdef'}
        files={n:hashlib.sha256(b).hexdigest()for n,b in self.data.items()};cp=hashlib.sha256(s.bootstrap.canonical(files)).hexdigest()
        urls={n:f'https://fixture.r2.cloudflarestorage.com/b/public/checkpoints/{cp}/{n}?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature='+('a'*64)for n in files}
        self.manifest=dict(epoch='test-1',start=time.time()-10,deadline=time.time()+100,transport_policy='direct-r2-v1',checkpoint=dict(id=cp,files=files,read_urls=urls))
        self.args=SimpleNamespace(authority='a'*64,key=str(self.root/'key'),state=str(self.root/'state'),source_cache=str(self.root/'source'),discovery_url='https://affine.io/mining.json',env_id='affine_math',max_batches=3,search_budget=32)
        self.meta=self.root/'meta';self.meta.mkdir();self.body,desc=archive();self.manifest['source_bundle']=desc
        lock=open(self.meta/'lock','a+b');self.addCleanup(lock.close);self.args.lock_fd=lock.fileno()
        self.observed=dict(manifest=self.manifest,manifest_url=next(iter(urls.values())))
        return self.manifest

    def test_interrupted_ranges_resume_and_final_hash_admission(self):
        m=self.fixture();session=Session(self.data,fail_once=True)
        dest=s.prefetch_checkpoint(m,self.args.state,session=session)
        for n,b in self.data.items():self.assertEqual((dest/n).read_bytes(),b)
        self.assertTrue(any(start==2 and end>2 for _,start,end in session.requests))
        second=Session(self.data);s.prefetch_checkpoint(m,self.args.state,session=second)
        self.assertTrue(all(end==0 for _,start,end in second.requests))

    def test_complete_bad_partial_never_becomes_model(self):
        m=self.fixture();stage=Path(self.args.state)/'.checkpoint-downloads'/m['checkpoint']['id'];stage.mkdir(parents=True)
        (stage/'model.safetensors.partial').write_bytes(b'x'*len(self.data['model.safetensors']))
        with self.assertRaises(ValueError):s.prefetch_checkpoint(m,self.args.state,session=Session(self.data))
        self.assertFalse((Path(self.args.state)/m['checkpoint']['id']/'model.safetensors').exists())
        self.assertEqual((stage/'model.safetensors.partial').read_bytes(),b'x'*16)

    def test_deadline_during_download_never_launches(self):
        self.fixture();launcher=Mock()
        def deadline(*args):
            self.manifest['deadline']=time.time()-1;raise ValueError('read plan expired')
        with patch.object(s.bootstrap,'download',return_value=self.body),patch.object(s.bootstrap,'hydrate_task_assets'):
            result=s.cycle(self.args,self.meta,read_opening=lambda *a:self.observed,prefetch=deadline,launcher=launcher)
        self.assertEqual(result,'closed-during-download');launcher.assert_not_called()

    def test_source_or_checkpoint_change_before_launch_refuses(self):
        self.fixture();changed=copy.deepcopy(self.observed);changed['manifest']['source_bundle']['sha256']='c'*64
        reads=iter([self.observed,changed]);launcher=Mock()
        with patch.object(s.bootstrap,'download',return_value=self.body),patch.object(s.bootstrap,'hydrate_task_assets'):
            result=s.cycle(self.args,self.meta,read_opening=lambda *a:next(reads),prefetch=Mock(),launcher=launcher)
        self.assertEqual(result,'changed-or-closed');launcher.assert_not_called()

    def test_original_source_once_and_no_duplicate_epoch_dispatch(self):
        self.fixture();child=Mock(pid=99999999);child.wait.return_value=0;popen=Mock(return_value=child)
        record=self.meta/'test-1.json'
        self.assertTrue(s.launch_once(self.args,self.observed,self.root,record,popen=popen))
        self.assertFalse(s.launch_once(self.args,self.observed,self.root,record,popen=popen));popen.assert_called_once()
        command=popen.call_args.args[0];self.assertIn('--manifest-url',command);self.assertIn('--once',command)
        self.assertIn(self.manifest['source_bundle']['sha256'],command)
        self.assertEqual(json.loads(record.read_bytes())['status'],'complete')

    def test_new_source_epoch_is_freshly_admitted(self):
        self.fixture();launched=[]
        def launch(args,observed,source,record):
            launched.append((observed['manifest']['source_bundle']['sha256'],str(source)))
            s.save(record,{'status':'complete'});return True
        with patch.object(s.bootstrap,'download',return_value=self.body),patch.object(s.bootstrap,'hydrate_task_assets'):
            self.assertEqual(s.cycle(self.args,self.meta,read_opening=lambda *a:self.observed,prefetch=Mock(),launcher=launch),'issued')
            self.assertEqual(s.cycle(self.args,self.meta,read_opening=lambda *a:self.observed,prefetch=Mock(),launcher=launch),'already-issued')
        self.body,desc=archive([('subnet/__init__.py',b''),('subnet/cli.py',b'print(1)'),(s.bootstrap.TASK_ASSET,b'[]')]);self.manifest.update(epoch='test-2',source_bundle=desc)
        with patch.object(s.bootstrap,'download',return_value=self.body),patch.object(s.bootstrap,'hydrate_task_assets'):
            self.assertEqual(s.cycle(self.args,self.meta,read_opening=lambda *a:self.observed,prefetch=Mock(),launcher=launch),'issued')
        self.assertEqual(len(launched),2);self.assertNotEqual(launched[0],launched[1])

    def test_old_launch_intent_blocks_other_epochs(self):
        self.fixture();s.save(self.meta/'old.json',{'status':'launching'});read=Mock()
        with self.assertRaises(ValueError):s.cycle(self.args,self.meta,read_opening=read)
        read.assert_not_called()

    def test_interrupted_inherited_launch_does_not_block_future_epochs(self):
        self.fixture();path=self.meta/'old.json'
        s.save(path,{'status':'launching','lock_inherited':True,'deadline':time.time()-10})
        self.assertFalse(s.original_child_pending(self.meta))
        self.assertEqual(json.loads(path.read_bytes())['status'],'interrupted')

    def test_child_inherits_lock_and_parent_crash_cannot_overlap(self):
        self.fixture();path=self.meta/'shared-lock'
        parent=open(path,'a+b');fcntl.flock(parent,fcntl.LOCK_EX|fcntl.LOCK_NB)
        child=subprocess.Popen([sys.executable,'-c','import sys;sys.stdin.read()'],stdin=subprocess.PIPE,pass_fds=(parent.fileno(),))
        self.addCleanup(lambda:child.poll() is None and child.kill())
        parent.close()
        with open(path,'a+b') as contender:
            with self.assertRaises(BlockingIOError):fcntl.flock(contender,fcntl.LOCK_EX|fcntl.LOCK_NB)
            child.communicate(timeout=5)
            fcntl.flock(contender,fcntl.LOCK_EX|fcntl.LOCK_NB)

    def test_new_verified_checkpoint_retires_only_owned_old_cache(self):
        self.fixture();old=s.prefetch_checkpoint(self.manifest,self.args.state,session=Session(self.data))
        unrelated=Path(self.args.state)/('f'*64);unrelated.mkdir();(unrelated/'keep').write_text('original')
        original=copy.deepcopy(self.manifest);self.data['config.json']=b'{"new":true}'
        files={n:hashlib.sha256(b).hexdigest()for n,b in self.data.items()};identifier=hashlib.sha256(s.bootstrap.canonical(files)).hexdigest()
        self.manifest['checkpoint']={'id':identifier,'files':files,'read_urls':{n:url.replace(original['checkpoint']['id'],identifier)for n,url in original['checkpoint']['read_urls'].items()}}
        new=s.prefetch_checkpoint(self.manifest,self.args.state,session=Session(self.data))
        self.assertFalse(old.exists());self.assertTrue(new.is_dir());self.assertEqual((unrelated/'keep').read_text(),'original')

    def test_bad_new_checkpoint_does_not_retire_previous_owned_cache(self):
        self.fixture();old=s.prefetch_checkpoint(self.manifest,self.args.state,session=Session(self.data))
        self.manifest['checkpoint']['files']['config.json']='f'*64
        with self.assertRaises(ValueError):s.prefetch_checkpoint(self.manifest,self.args.state,session=Session(self.data))
        self.assertTrue(old.is_dir())

    def test_transient_discovery_error_can_retry_without_weakening_signatures(self):
        self.fixture()
        import requests
        with self.assertRaises(requests.HTTPError):
            s.opening(self.args.discovery_url,self.args.authority,get=lambda *a,**k:Response(b'',status=503))
        hint={'accepting_submissions':False}
        self.assertIsNone(s.opening(self.args.discovery_url,self.args.authority,get=lambda *a,**k:Response(json.dumps(hint).encode(),status=200)))

    def test_packaged_transfer_matches_full_shard_qualified_operator_functions(self):
        import inspect
        from ops import checkpoint_read_hydration as original
        for name in ('download','read_url','sha'):
            self.assertEqual(inspect.getsource(getattr(s.hydration,name)),inspect.getsource(getattr(original,name)))

    def test_compressed_discovery_is_decoded_and_bounded(self):
        self.fixture()
        import gzip
        body=json.dumps({'accepting_submissions':False}).encode()
        response=Response(body,headers={'Content-Encoding':'gzip'},status=200)
        response.raw=io.BytesIO(gzip.compress(body))
        self.assertIsNone(s.opening(self.args.discovery_url,self.args.authority,get=lambda *a,**k:response))
        response=Response(b' '*(s.bootstrap.JSON_LIMIT+1),status=200)
        with self.assertRaisesRegex(ValueError,'size bound'):
            s.opening(self.args.discovery_url,self.args.authority,get=lambda *a,**k:response)

    def test_incomplete_or_proxy_discovery_is_retryable_but_authority_is_not(self):
        self.fixture()
        import requests
        for body in (b'{"accepting',b'<html>proxy</html>',b'[]',b'\xff'):
            with self.assertRaises(requests.HTTPError):
                s.opening(self.args.discovery_url,self.args.authority,get=lambda *a,**k:Response(body,status=200))
        body=json.dumps({'accepting_submissions':True,'authority':'b'*64}).encode()
        with self.assertRaisesRegex(ValueError,'authority'):
            s.opening(self.args.discovery_url,self.args.authority,get=lambda *a,**k:Response(body,status=200))

    def test_checkpoint_throttling_retries_without_admitting_wrong_bytes(self):
        m=self.fixture();normal=Session(self.data);count=[0]
        class Throttled:
            def get(inner,url,headers=None,**kwargs):
                if headers['Range']!='bytes=0-0' and count[0]==0:
                    count[0]+=1;return Response(b'',status=429)
                return normal.get(url,headers=headers,**kwargs)
            def close(inner):pass
        dest=s.prefetch_checkpoint(m,self.args.state,session=Throttled())
        self.assertEqual(count[0],1)
        self.assertEqual((dest/'model.safetensors').read_bytes(),self.data['model.safetensors'])

    def test_signatures_required_and_expired_discovery_never_opens(self):
        self.fixture();key=SigningKey.generate();authority=key.verify_key.encode().hex()
        def sign(body):return s.bootstrap.canonical(dict(payload=body,signer=authority,signature=base64.b64encode(key.sign(s.bootstrap.canonical(body)).signature).decode()))
        hint=dict(authority=authority,accepting_submissions=True,epoch='test-1',current_url=self.observed['manifest_url'])
        pointer=dict(transport_policy='direct-r2-v1',epoch='test-1',manifest_url=self.observed['manifest_url'])
        def read(*args,**kwargs):return Response(json.dumps(hint).encode(),status=200)
        responses=iter([sign(pointer),sign(self.manifest)])
        self.assertEqual(s.opening(self.args.discovery_url,authority,get=read,fetch=lambda *a:next(responses))['manifest'],self.manifest)
        bad=json.loads(sign(self.manifest));bad['payload']['checkpoint']['id']='c'*64
        responses=iter([sign(pointer),json.dumps(bad).encode()])
        with self.assertRaises(Exception):s.opening(self.args.discovery_url,authority,get=read,fetch=lambda *a:next(responses))
        self.manifest['deadline']=time.time()-1;responses=iter([sign(pointer),sign(self.manifest)])
        self.assertIsNone(s.opening(self.args.discovery_url,authority,get=read,fetch=lambda *a:next(responses)))
