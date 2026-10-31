"""Real Miner, Ed25519, ZIP and admission; all HTTP/S3 are in-memory only."""
from test_complete_group_fixtures import *
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import io
import threading
import requests
from urllib.parse import urlparse,unquote,parse_qs
import zipfile
from botocore.exceptions import ClientError
from subnet.storage import Identity
from subnet.complete_group_codec import limits,encode,decode,loads
from subnet.complete_group_admission import (TaskAuthority,context,scope,group_id,document,signed_commitment,
    validate_commitment,structural_batch,validate_document,TRANSPORT)
from subnet.complete_group_transport import CompleteGroupTransport,capability,paths,validate_capability
from subnet.complete_group_capture import AdmissionLedger,capture


class FakeBucket:
    def __init__(self):
        self.name='candidate';self.client=self;self.objects={};self.reads=[];self.heads=[];self.puts=[]
        self.clock=time.time();self.lock=threading.Lock();self.fail_snapshot=False
    def presign(self,key,method,expiry):
        assert method in ('put_object','get_object') and expiry>0
        return 'https://test.r2.cloudflarestorage.com/candidate/'+key+'?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=mock'
    def generate_presigned_url(self,method,*,Params,ExpiresIn):
        # Real botocore presigner, fake credentials, no network request.
        import boto3
        from botocore.config import Config
        signer=boto3.client('s3',endpoint_url='https://test.r2.cloudflarestorage.com',
            aws_access_key_id='FAKE',aws_secret_access_key='FAKE',region_name='auto',config=Config(signature_version='s3v4'))
        return signer.generate_presigned_url(method,Params=Params,ExpiresIn=ExpiresIn)
    def store(self,key,body):
        self.clock+=.001;self.objects[key]=(bytes(body),self.clock)
    def put_object(self,*,Bucket,Key,Body,IfNoneMatch=None,**kwargs):
        with self.lock:
            if self.fail_snapshot and '/complete-group-snapshots/' in Key:
                self.fail_snapshot=False;raise OSError('mock snapshot failure')
            if IfNoneMatch=='*' and Key in self.objects:
                raise ClientError({'ResponseMetadata':{'HTTPStatusCode':412},'Error':{'Code':'PreconditionFailed'}},'PutObject')
            self.puts.append(Key);self.store(Key,Body)
        return {}
    def get_object(self,*,Bucket,Key):
        with self.lock:
            data,stamp=self.objects[Key];self.reads.append(Key)
        return dict(Body=io.BytesIO(data),ContentLength=len(data),LastModified=datetime.fromtimestamp(stamp,timezone.utc))
    def head_object(self,*,Bucket,Key):
        with self.lock:
            data,stamp=self.objects[Key];self.heads.append(Key)
        return dict(ContentLength=len(data),LastModified=datetime.fromtimestamp(stamp,timezone.utc))


class FakeHTTP:
    def __init__(self,bucket):self.bucket=bucket;self.puts=[];self.gets=[];self.fail=None;self.redirect=False;self.lose_ack=False
    def put(self,url,*,data,headers,timeout,allow_redirects):
        assert headers.get('Content-Type')=='application/octet-stream' and 0<timeout<=120 and allow_redirects is False
        key=unquote(urlparse(url).path).removeprefix('/candidate/')
        self.puts.append((key,bytes(data)))
        if self.fail and self.fail in key:
            self.fail=None;raise requests.ConnectionError('mock interrupted PUT')
        if self.redirect:return SimpleNamespace(status_code=307)
        signed=parse_qs(urlparse(url).query).get('X-Amz-SignedHeaders',[''])[0].split(';')
        if 'if-none-match' in signed and headers.get('If-None-Match')!='*':return SimpleNamespace(status_code=403)
        try:self.bucket.put_object(Bucket=self.bucket.name,Key=key,Body=data,IfNoneMatch=headers.get('If-None-Match'))
        except ClientError:return SimpleNamespace(status_code=412)
        if self.lose_ack:
            self.lose_ack=False;raise requests.ConnectionError('mock write succeeded but acknowledgement lost')
        return SimpleNamespace(status_code=200,raise_for_status=lambda:None)
    def get(self,url,*,stream,timeout,allow_redirects):
        assert stream is True and allow_redirects is False and 0<timeout<=120
        key=unquote(urlparse(url).path).removeprefix('/candidate/');self.gets.append(key)
        data,stamp=self.bucket.objects[key]
        return SimpleNamespace(status_code=200,headers={'Content-Length':str(len(data))},
            iter_content=lambda chunk_size:iter([data]),close=lambda:None)


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.state=Path(self.tmp.name)/'miner.json';self.ledger=AdmissionLedger(Path(self.tmp.name)/'admission.sqlite')
        self.identity=Identity(bytes(range(32)));self.other=Identity(bytes(reversed(range(32))))
        self.bucket=FakeBucket();self.http=FakeHTTP(self.bucket);self.actors=[]
        self.addCleanup(lambda:[a._group_journal.close() for a in self.actors])
    def manifest(self,n=8):
        m=fixture(n);encrypted=capability(self.bucket,m,self.identity.id,registered_miners={self.identity.id})
        m['capabilities']={self.identity.id:encrypted};return m
    def actor(self,m,state=None,automatic=False):
        with patch('subnet.model.check_runtime_profile'):
            a=Miner(self.identity,m,'no-model-loaded',state_path=state,
                **({} if automatic else {'complete_group_transport':CompleteGroupTransport(http=self.http)}))
        if automatic:a._group_transport.http=self.http
        self.actors.append(a);return a
    def search(self,a,index=0,outcomes=None,error=None):
        r=FixtureRuntime(a.manifest,self.identity.id,outcomes,error)
        with patch('subnet.miner.make_runtime',return_value=r):batch=a.search(index,env_id='math')
        return batch,r
    def resolve(self,m,env,index):
        return TaskAuthority(m['epoch'],m['checkpoint']['id'],env,'taskset-1',index,
            digest(dict(index=index,taskset='taskset-1')),(1,index+2),15,16)
    def capture(self,m,**kwargs):
        return capture(self.bucket,self.ledger,m,self.identity.id,registered_miners={self.identity.id},
            resolve_task=kwargs.pop('resolve_task',self.resolve),now=self.bucket.clock+1,**kwargs)
    def test_real_miner_auto_dispatch_capability_full_group_and_private_capture(self):
        m=self.manifest();a=self.actor(m,self.state,automatic=True);batch,r=self.search(a)
        self.assertEqual(len(r.calls),8);self.assertEqual([x['seed'] for x in batch['rollouts']],list(range(8)))
        original,arrays=decode(a._group_prepared[0][1],limits(m))[0]
        self.assertEqual(original,batch);self.assertEqual(len(arrays),8)
        self.assertEqual(batch['rollouts'][2]['turns'][0]['output'],batch['rollouts'][3]['turns'][0]['output'])
        self.assertEqual(a.upload(),200);result=self.capture(m)
        self.assertEqual(result['new_groups'],1);self.assertFalse(result['fully_audited'])
        self.assertEqual(result['proof_status'],'declared-not-fully-downloaded-or-verified')
        self.assertTrue(all(k.startswith('private/') for k in self.bucket.objects))
        self.assertTrue(any('/staging/' in k for k in self.bucket.heads))
        self.assertFalse(any('/staging/' in k for k in self.bucket.reads))
    def test_retry_skips_acknowledged_artifact_and_publishes_commitment_last(self):
        m=self.manifest();a=self.actor(m,self.state);self.search(a);self.http.fail='/training/'
        with self.assertRaises(requests.ConnectionError):a.upload()
        self.assertNotIn(paths(m,self.identity.id),self.bucket.objects)
        self.assertEqual(a.upload(),200)
        keys=[k for k,_ in self.http.puts];self.assertEqual(sum('/staging/' in k for k in keys),1)
        self.assertEqual(keys[-1],paths(m,self.identity.id));self.assertEqual(self.capture(m)['new_groups'],1)
        self.assertEqual(a.upload(),200);self.assertEqual(self.capture(m)['new_groups'],0)
    def test_restart_restores_exact_prepared_groups_and_no_model_reexecution(self):
        m=self.manifest();a=self.actor(m,self.state);original,_=self.search(a);data=a._group_prepared[0][1]
        a.upload();a._group_journal.close();b=self.actor(m,self.state)
        with patch('subnet.miner.make_runtime',side_effect=AssertionError('no regeneration')):
            self.assertEqual(b.search(0,env_id='math'),original)
        self.assertEqual(b._group_prepared[0][1],data);self.assertEqual(b.upload(),200)
        self.assertEqual(sum('/staging/' in k for k,_ in self.http.puts),1)
    def test_original_interrupted_prefix_resumes_through_new_codec(self):
        m=self.manifest();a=self.actor(m,self.state)
        with self.assertRaisesRegex(RuntimeError,'interrupted'):self.search(a,error=3)
        a._group_journal.close();b=self.actor(m,self.state);batch,r=self.search(b)
        self.assertEqual(r.calls,[(0,i) for i in range(3,8)]);self.assertEqual(len(batch['rollouts']),8)
    def test_cap_applies_cumulatively_and_existing_task_is_idempotent(self):
        m=self.manifest();a=self.actor(m,self.state)
        for index in range(3):self.search(a,index);a.upload();self.assertEqual(self.capture(m)['new_groups'],1)
        self.assertEqual(len(a.batches),3)
        with patch('subnet.miner.make_runtime',side_effect=AssertionError('cap before compute')):
            self.assertEqual(a.search(0,env_id='math')['index'],0)
            with self.assertRaisesRegex(ValueError,'cap'):a.search(3,env_id='math')
    def test_wrong_capability_or_identity_fails_before_model(self):
        m=self.manifest();cap=self.identity.decrypt(m['capabilities'][self.identity.id]);bad=deepcopy(cap)
        bad['batch_put_urls'][0]=bad['batch_put_urls'][1]
        with self.assertRaises(ValueError):validate_capability(bad,m,self.identity.id)
        with self.assertRaises(Exception):self.other.decrypt(m['capabilities'][self.identity.id])
        with self.assertRaises(ValueError):capability(self.bucket,m,self.other.id,registered_miners={self.identity.id})
        for change in ('scope','version','extra','host'):
            bad=deepcopy(cap)
            if change=='scope':bad['scope_sha256']='0'*64
            if change=='version':bad['transport']='small-commitment-pairs-v2'
            if change=='extra':bad['credentials']='forbidden'
            if change=='host':bad['put_url']=bad['put_url'].replace('test.r2','foreign.r2')
            with self.assertRaises(ValueError):validate_capability(bad,m,self.identity.id)
    def test_signed_commitment_authorization_unknown_fields_and_stale_binding(self):
        m=self.manifest();a=self.actor(m);self.search(a);raw=signed_commitment(self.identity,m,a._group_prepared)
        with self.assertRaises(ValueError):validate_commitment(raw,m,self.identity.id,registered_miners={self.other.id})
        for field,value in [('epoch','other'),('source','1'*64),('version','legacy'),('unexpected',1)]:
            envelope=json.loads(raw);envelope['payload'][field]=value
            envelope['signature']=base64.b64encode(self.identity.key.sign(canonical(envelope['payload'])).signature).decode()
            with self.assertRaises(ValueError):validate_commitment(canonical(envelope),m,self.identity.id,registered_miners={self.identity.id})
        bad=bytearray(raw);bad[-5]^=1
        with self.assertRaises(Exception):validate_commitment(bytes(bad),m,self.identity.id,registered_miners={self.identity.id})
    def test_altered_prepared_bytes_and_metadata_are_rejected(self):
        m=self.manifest();a=self.actor(m,self.state);self.search(a);batch,data=a._group_prepared[0]
        altered=deepcopy(batch);altered['rollouts'][0]['turns'][0]['text']='new text'
        with self.assertRaisesRegex(ValueError,'metadata'):a._group_transport.persist(m,self.identity.id,self.state,[(altered,data)])
        changed=data[:-1]+bytes([data[-1]^1])
        with self.assertRaises(Exception):a._group_transport.persist(m,self.identity.id,self.state,[(batch,changed)])
        p=next(Path(str(self.state)+'.complete-groups').glob('*.zip'));p.write_bytes(data+b'x');a._group_journal.close()
        with self.assertRaises(ValueError):self.actor(m,self.state)
    def test_structural_rejects_missing_reordered_unknown_and_changed_attempts(self):
        m=self.manifest();a=self.actor(m);batch,_=self.search(a);task=self.resolve(m,'math',0)
        mutations=[lambda b:b['rollouts'].pop(),lambda b:b['rollouts'].reverse(),
            lambda b:b['rollouts'][0].update(seed=1),lambda b:b.update(extra=True),
            lambda b:b['rollouts'][0]['turns'][0].update(extra=True),
            lambda b:b['rollouts'][0]['turns'][0].update(proofs=['wrong']),
            lambda b:b['rollouts'][0]['turns'][0]['behavior_sampling'].update(attempt=2)]
        for mutate in mutations:
            bad=deepcopy(batch);mutate(bad)
            with self.assertRaises(ValueError):structural_batch(bad,m,self.identity.id,task)
    def test_native_authority_rejects_task_hash_prompt_or_tokenizer_override(self):
        m=self.manifest();a=self.actor(m);self.search(a);a.upload()
        for key,value in [('task_hash','f'*64),('prompt',(1,9)),('eos_token_id',14),('vocab_size',18)]:
            def resolver(manifest,env,index):
                d=self.resolve(manifest,env,index).__dict__.copy();d[key]=value;return TaskAuthority(**d)
            with self.assertRaises(ValueError):self.capture(m,resolve_task=resolver)
        self.assertEqual(self.capture(m)['new_groups'],1)
    def test_budget_output_is_retained_neutral_and_selected_lp_is_not_behavior_lp(self):
        m=self.manifest();a=self.actor(m);batch,_=self.search(a,outcomes=['positive','negative','budget-positive']+['neutral']*5)
        _,arrays=decode(a._group_prepared[0][1],limits(m))[0];row=batch['rollouts'][2]
        self.assertEqual(row['classification'],'neutral');self.assertEqual(row['native_classification'],'positive')
        self.assertEqual(len(row['turns'][0]['output']),12)
        self.assertTrue((arrays[2][0]==-2).all());self.assertEqual(row['turns'][0]['behavior_sampling']['behavior_selected_logprobs'],[-1.]*12)
        a.upload();self.assertEqual(self.capture(m)['new_groups'],1)
    def test_compact_document_mutation_does_not_enter_ledger(self):
        m=self.manifest();a=self.actor(m);self.search(a);a.upload();key=paths(m,self.identity.id,0)[1]
        body,stamp=self.bucket.objects[key];self.bucket.objects[key]=(body.replace(b'fixture',b'changed'),stamp)
        with self.assertRaises(ValueError):self.capture(m)
        self.bucket.objects[key]=(body,stamp);self.assertEqual(self.capture(m)['new_groups'],1)
    def test_cumulative_commitment_cannot_replace_or_regress_existing_credit(self):
        m=self.manifest();a=self.actor(m);self.search(a,0);a.upload();self.capture(m)
        first=list(a._group_prepared);self.search(a,1);a.upload();self.assertEqual(self.capture(m)['new_groups'],1)
        # Even a correctly re-signed old commitment cannot retract a known slot.
        old=signed_commitment(self.identity,m,first);self.bucket.store(paths(m,self.identity.id),old)
        with self.assertRaisesRegex(ValueError,'regression'):self.capture(m)
    def test_concurrent_capture_counts_the_same_group_exactly_once(self):
        m=self.manifest();a=self.actor(m);self.search(a);a.upload()
        with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(lambda _:self.capture(m),range(2)))
        self.assertEqual(sorted(r['new_groups'] for r in results),[0,1])
    def test_snapshot_failure_does_not_credit_then_retries_idempotently(self):
        m=self.manifest();a=self.actor(m);self.search(a);a.upload();self.bucket.fail_snapshot=True
        with self.assertRaises(OSError):self.capture(m)
        self.assertEqual(self.capture(m)['new_groups'],1);self.assertEqual(self.capture(m)['new_groups'],0)
    def test_stale_or_late_objects_rejected_and_redirect_not_acknowledged(self):
        m=self.manifest();a=self.actor(m);self.search(a);self.http.redirect=True
        with self.assertRaises(ValueError):a.upload()
        self.http.redirect=False;a.upload();key=paths(m,self.identity.id);body,stamp=self.bucket.objects[key]
        self.bucket.objects[key]=(body,m['deadline'])
        with self.assertRaises(ValueError):self.capture(m)
        self.bucket.objects[key]=(body,stamp);proof=paths(m,self.identity.id,0)[0];data,_=self.bucket.objects[proof]
        self.bucket.objects[proof]=(data,stamp+1)
        with self.assertRaises(ValueError):self.capture(m)

    def test_actual_gateway_opener_persists_new_epoch_and_encrypted_capabilities(self):
        from subnet.storage import Gateway
        from subnet.complete_group_transport import open_epoch
        m=fixture();state=Path(self.tmp.name)/'gateway.json'
        gateway=Gateway(self.bucket,state_path=state,direct_r2=True)
        try:
            caps=open_epoch(gateway,m,{self.identity.id})
            cap=self.identity.decrypt(caps[self.identity.id]);validate_capability(cap,m,self.identity.id)
            saved=json.loads(state.read_bytes())
            self.assertEqual(saved['epochs'][m['epoch']]['commitment_binding']['version'],TRANSPORT)
            self.assertEqual(saved['epochs'][m['epoch']]['miners'],[self.identity.id])
            with self.assertRaises(ValueError):open_epoch(gateway,m,{self.identity.id})
        finally:gateway.stop()
    def test_immutable_slot_retry_confirms_original_full_bytes_after_lost_ack(self):
        m=self.manifest();a=self.actor(m,self.state);self.search(a);self.http.lose_ack=True
        with self.assertRaises(requests.ConnectionError):a.upload()
        self.assertNotIn(paths(m,self.identity.id),self.bucket.objects)
        self.assertEqual(a.upload(),200);self.assertEqual(len(self.http.gets),1)
        self.assertEqual(self.capture(m)['new_groups'],1)
        cap=a.cap;url=cap['batch_put_urls'][0];before=self.bucket.objects[paths(m,self.identity.id,0)[0]][0]
        response=self.http.put(url,data=b'overwrite',headers=cap['upload_headers'],timeout=10,allow_redirects=False)
        self.assertEqual(response.status_code,412);self.assertEqual(self.bucket.objects[paths(m,self.identity.id,0)[0]][0],before)
    def test_immutable_slot_different_body_and_missing_condition_fail_closed(self):
        m=self.manifest();a=self.actor(m,self.state);self.search(a);url=a.cap['batch_put_urls'][0]
        response=self.http.put(url,data=b'wrong',headers=a.cap['headers'],timeout=10,allow_redirects=False)
        self.assertEqual(response.status_code,403)
        bad=deepcopy(a.cap);bad['batch_put_urls'][0]=self.bucket.presign(paths(m,self.identity.id,0)[0],'put_object',30)
        with self.assertRaisesRegex(ValueError,'signature-bound'):validate_capability(bad,m,self.identity.id)
        self.bucket.store(paths(m,self.identity.id,0)[0],b'pre-existing wrong original')
        with self.assertRaisesRegex(ValueError,'size differs'):a.upload()
        body=a._group_prepared[0][1];self.bucket.store(paths(m,self.identity.id,0)[0],bytes([body[0]^1])+body[1:])
        with self.assertRaisesRegex(ValueError,'bytes differ'):a.upload()
        self.assertNotIn(paths(m,self.identity.id),self.bucket.objects)
    def test_auditor_artifact_reader_requires_exact_committed_bytes_and_document(self):
        from subnet.complete_group_admission import validate_artifact
        m=self.manifest();a=self.actor(m);batch,_=self.search(a);data=a._group_prepared[0][1]
        commitment=json.loads(signed_commitment(self.identity,m,a._group_prepared))
        claim=commitment['payload']['groups'][0];body=document(batch,m,self.identity.id,0)
        result,arrays,assessment=validate_artifact(data,m,self.identity.id,claim,self.resolve(m,'math',0),expected_document=body)
        self.assertEqual(result,batch);self.assertEqual(len(arrays),8);self.assertFalse(assessment['sampling_verified'])
        with self.assertRaises(ValueError):validate_artifact(data+b'bad',m,self.identity.id,claim,self.resolve(m,'math',0),expected_document=body)
        changed=deepcopy(batch);changed['rollouts'][0]['turns'][0]['text']='altered'
        other=encode([(changed,arrays)],limits(m));altered=deepcopy(claim);altered['artifact_sha256']=hashlib.sha256(other).hexdigest();altered['artifact_size']=len(other)
        with self.assertRaisesRegex(ValueError,'same original group'):
            validate_artifact(other,m,self.identity.id,altered,self.resolve(m,'math',0),expected_document=body)
    def test_padded_head_is_allowed_but_unsupported_tokenizer_is_rejected(self):
        m=self.manifest();adapter=CompleteGroupTransport();runtime=FixtureRuntime(m,self.identity.id)
        runtime.model.config.vocab_size=32;adapter.check_runtime(runtime,m)
        runtime.model.config.vocab_size=15
        with self.assertRaises(ValueError):adapter.check_runtime(runtime,m)
        runtime.model.config.vocab_size=32;runtime.tokenizer.eos_token_id=14
        with self.assertRaises(ValueError):adapter.check_runtime(runtime,m)


class CodecTests(unittest.TestCase):
    def test_new_codec_preserves_64_rollouts_with_4096_rows(self):
        m=fixture(64);m['max_batches']=1;m['group_transport_policy']['max_context_tokens']=8192
        m['environments'][0]['harness']['max_output_tokens']=4096
        bound=limits(m);arrays=[[np.full((4096,1),-2.,np.float32)] for _ in range(64)]
        data=encode([({'original_test_vectors':True},arrays)],bound)
        row,restored=decode(data,bound)[0];self.assertEqual(len(restored),64)
        self.assertEqual(row,{'original_test_vectors':True});self.assertTrue(np.array_equal(restored[63][0],arrays[63][0]))
    def test_caps_and_nonfinite_arrays_fail_closed(self):
        m=fixture();bound=limits(m)
        for array in (np.zeros((13,1),np.float32),np.zeros((2,16),np.float32),np.ones((2,1),np.float32),np.full((2,1),np.nan,np.float32)):
            with self.assertRaises(ValueError):encode([({},[[array]])],bound)
        m['group_contract']['group_size']=129
        with self.assertRaises(ValueError):limits(m)
    def test_zip_directory_count_rejected_before_zipinfo_allocation(self):
        import struct
        bound=limits(fixture());data=encode([({},[[np.zeros((2,1),np.float32)]])],bound)
        fields=list(struct.unpack('<4s4H2LH',data[-22:]));fields[3]=fields[4]=60000
        bad=data[:-22]+struct.pack('<4s4H2LH',*fields)
        with patch('subnet.complete_group_codec.zipfile.ZipFile',side_effect=AssertionError('must reject before ZipInfo allocation')):
            with self.assertRaisesRegex(ValueError,'before allocation'):decode(bad,bound)
    def test_unknown_zip_members_duplicate_json_keys_and_declared_shape_rejected(self):
        bound=limits(fixture());data=encode([({},[[np.zeros((2,1),np.float32)]])],bound)
        source=zipfile.ZipFile(io.BytesIO(data));members={x:source.read(x) for x in source.namelist()}
        for change in ('extra','duplicate-json','shape'):
            modified=dict(members)
            if change=='extra':modified['../../secret']=b'x'
            if change=='duplicate-json':modified['manifest.json']=b'[{"batch":{},"batch":{},"arrays":[["0.npy"]]}]'
            if change=='shape':modified['0.npy']=members['0.npy'].replace(b'(2, 1)',b'(9, 1)')
            output=io.BytesIO()
            with zipfile.ZipFile(output,'w') as z:
                for name,body in modified.items():z.writestr(name,body)
            with self.assertRaises(ValueError):decode(output.getvalue(),bound)

if __name__=='__main__':unittest.main()
