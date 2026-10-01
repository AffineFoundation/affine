"""Protocol invariants independent of costly model execution."""
import tempfile
import hashlib
import time
import unittest
from pathlib import Path
import requests
from .storage import Gateway,Identity
from .scoring import score
from .client import checkpoint_download

class MemoryBucket:
    def __init__(self):self.objects={}
    def put(self,key,data,*_):self.objects[key]=data
    def get(self,key):return self.objects[key]
    def json(self,key,value):self.put(key,str(value).encode())

class ProtocolTests(unittest.TestCase):
    def test_invalid_cannot_cancel_valid(self):
        result=score({'a':{'accepted':[{'index':1},{'index':2}]},'b':{'accepted':[{'index':2}]},'c':{'accepted':[]}})
        self.assertEqual(result['points'],{'a':1,'b':0,'c':0})
        self.assertEqual(score({'a':{'accepted':[]}})['weights'],{})

    def test_pinned_checkpoint_cache_and_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            bucket=MemoryBucket();gateway=Gateway(bucket)
            good=b'authority-pinned tokenizer';digest=hashlib.sha256(good).hexdigest()
            bucket.put('public/checkpoints/a/tokenizer.json',good)
            m={'checkpoint':{'files':{'tokenizer.json':digest},'base_url':gateway.url+'/public/checkpoints/a'}}
            first=checkpoint_download(m,Path(d)/'a')
            m['checkpoint']['base_url']=gateway.url+'/public/checkpoints/b'
            second=checkpoint_download(m,Path(d)/'b')
            self.assertEqual((second/'tokenizer.json').read_bytes(),good)
            # Mutating a cached inode must never bypass the next pinned digest.
            (first/'tokenizer.json').write_bytes(b'corrupt cache')
            bucket.put('public/checkpoints/b/tokenizer.json',good)
            repaired=checkpoint_download(m,second)
            self.assertEqual((repaired/'tokenizer.json').read_bytes(),good)
            m['checkpoint']['base_url']=gateway.url+'/public/checkpoints/c'
            bucket.put('public/checkpoints/c/tokenizer.json',b'wrong network bytes')
            m['checkpoint']['files']['tokenizer.json']=hashlib.sha256(b'unavailable exact bytes').hexdigest()
            with self.assertRaisesRegex(ValueError,'checkpoint integrity'):
                checkpoint_download(m,Path(d)/'c')
            self.assertFalse((Path(d)/'c/tokenizer.json').exists())
            self.assertFalse((Path(d)/'c/tokenizer.json.tmp').exists())
            gateway.stop()

    def test_capabilities_close_and_restart(self):
        with tempfile.TemporaryDirectory() as d:
            bucket=MemoryBucket();path=Path(d)/'gateway.json';key=Identity();other=Identity()
            gateway=Gateway(bucket,state_path=path)
            caps=gateway.open('test',[key.id],int(time.time())+60)
            with self.assertRaises(Exception):other.decrypt(caps[key.id])
            cap=key.decrypt(caps[key.id]);url=cap['put_url']
            self.assertEqual(requests.put(url,data=b'first',timeout=5).status_code,200)
            self.assertEqual(requests.get(gateway.url+'/private/test/x',timeout=5).status_code,403)
            self.assertEqual(requests.put(url.replace('signature=','signature=x'),data=b'bad',timeout=5).status_code,403)
            gateway.stop();gateway=Gateway(bucket,state_path=path)
            fresh=url.replace(url.split('/upload/')[0],gateway.url)
            self.assertEqual(requests.put(fresh,data=b'final',timeout=5).status_code,200)
            receipts=gateway.freeze('test')
            self.assertEqual(requests.put(fresh,data=b'late',timeout=5).status_code,403)
            self.assertEqual(bucket.get(receipts[key.id]['frozen_key']),b'final')
            gateway.stop()


class GenericProtocolTests(unittest.TestCase):
    def test_uniqueness_scoped_by_environment_and_checkpoint(self):
        first=dict(env_id='copy',index=1,sample_index=1,checkpoint='a')
        other=dict(env_id='math',index=1,sample_index=1,checkpoint='a')
        result=score({'a':{'accepted':[first,other]},'b':{'accepted':[first]}})
        self.assertEqual(result['points'],{'a':1,'b':0})
        next_version=dict(first,checkpoint='b')
        result=score({'a':{'accepted':[first]},'b':{'accepted':[next_version]}})
        self.assertEqual(result['points'],{'a':1,'b':1})

    def test_post_freeze_audit_sampling_and_assurance(self):
        from .auditing import select,assurance
        policy={'mode':'sampled','count':3}
        selected=select(8,policy,'1'*64,'receipt-a')
        self.assertEqual(selected,select(8,policy,'1'*64,'receipt-a'))
        self.assertEqual(len(set(selected)),3)
        self.assertNotEqual(selected,select(8,policy,'2'*64,'receipt-a'))
        self.assertEqual(select(8,{'mode':'full'},None,'x'),list(range(8)))
        with self.assertRaises(ValueError):select(8,policy,'g'*64,'x')
        self.assertAlmostEqual(assurance(10,2)['detection_probability'],.2)

    def test_sampled_scores_expose_unchecked_collision_and_training_boundary(self):
        batch=dict(env_id='math',index=2,sample_index=2,checkpoint='approved')
        report={'accepted':[batch],'outcomes':[{'batch':0,'fully_audited':True},{'batch':1,'fully_audited':False,'valid':None}]}
        result=score({'checked':report,'unchecked':{'accepted':[],'outcomes':[{'fully_audited':False,'valid':None}]}})
        self.assertTrue(result['provisional'])
        self.assertEqual(result['score_basis'],'fully-audited-subset')
        self.assertEqual(result['duplicate_coverage'],'incomplete')
        self.assertTrue(result['unchecked_duplicate_claims_unresolved'])
        self.assertEqual(result['points']['unchecked'],0)
        self.assertEqual(len(report['accepted']),1)

    def test_harness_tools_and_explicit_protocols(self):
        from .harness import action,normalize,observations,plain_render
        self.assertEqual(action('{"tool_call":{"name":"sum","arguments":{"x":2}}}')['tool_calls'][0]['name'],'sum')
        self.assertEqual(action('ordinary final answer')['tool_calls'],[])
        self.assertEqual(observations([{'role':'tool','content':'3'}],{})[0]['content'],'Tool result: 3')
        self.assertIn('ASSISTANT:',plain_render([{'role':'user','content':'hi'}]))
        with self.assertRaises(ValueError):normalize({'version':'uploaded-harness'})
        with self.assertRaises(ValueError):observations([{'role':'developer','content':'x'}],{})

    def test_harness_registry_dispatch_and_turn_policies(self):
        from . import harness
        class Tokenizer:
            def encode(self,text,**kwargs):return [len(text)]
            def apply_chat_template(self,messages,**kwargs):return {'input_ids':[17]}
        messages=[{'role':'user','content':'public request'}]
        self.assertEqual(harness.render(Tokenizer(),messages,config={'version':'text-tools-v1'}),[17])
        self.assertNotEqual(harness.render(Tokenizer(),messages,config={'version':'plain-transcript-v1'}),[17])
        config=dict(policy='autoregressive',max_output_tokens=32,turn_overrides={'0':{'policy':'candidates','candidates':['safe tool action','wrong arguments']}})
        self.assertEqual(harness.turn_config(config,0)['policy'],'candidates')
        self.assertEqual(harness.turn_config(config,1)['policy'],'autoregressive')
        for override in ({'max_output_tokens':33},{'version':'plain-transcript-v1'},{'hidden_answer':'oracle'}):
            with self.assertRaises(ValueError):harness.normalize(dict(config,turn_overrides={'0':override}))
        with self.assertRaises(ValueError):harness.normalize(dict(config,turn_overrides={'32':{}}))




class ServiceContractTests(unittest.TestCase):
    def test_multi_environment_signed_contract_and_heldout_separation(self):
        import os
        from unittest.mock import patch
        from .service import open_contract,DEFAULT_PROFILE
        config={'environments':[{'source':'mastermind','indices':[0,1]},{'source':'affine_verbatim','config':{'taskset':{'num_samples':4,'target_length':8,'content_type':'codes'}},'indices':[0,1],'harness':{'version':'plain-transcript-v1','policy':'autoregressive','max_output_tokens':16}}],
                'duration':1800,'audit_policy':{'mode':'sampled','count':1},'evaluation':{'suites':[{'env_id':'mastermind','indices':[2,3]},{'env_id':'affine_verbatim','indices':[2,3]}]}}
        with patch.dict(os.environ,DEFAULT_PROFILE):contract=open_contract(config)
        self.assertEqual(len(contract['environments']),2)
        self.assertEqual(contract['duration'],1800)
        self.assertEqual(contract['runtime_profile'],DEFAULT_PROFILE)
        self.assertEqual(contract['audit_policy']['mode'],'sampled')
        self.assertEqual(len(contract['evaluation']['suites']),2)
        config['evaluation']['suites'][0]['indices']=[0]
        with patch.dict(os.environ,DEFAULT_PROFILE),self.assertRaises(ValueError):open_contract(config)

    def test_checkpoint_disk_guard_holds_before_training(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from .controller import require_checkpoint_space,CheckpointCapacityError
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'model.safetensors').write_bytes(b'trusted test weights')
            with patch('shutil.disk_usage',return_value=SimpleNamespace(free=1024)):
                with self.assertRaises(CheckpointCapacityError):require_checkpoint_space(root,root/'new')
            with patch('shutil.disk_usage',return_value=SimpleNamespace(free=4*1024**3)):
                result=require_checkpoint_space(root,root/'new')
                self.assertEqual(result['required_bytes'],2*1024**3)

    def test_production_miner_retries_epoch_without_reloading_model(self):
        import sys
        from unittest.mock import patch
        from types import SimpleNamespace
        from . import cli
        events={'search':0,'loads':0,'sleep':0,'upload':0}
        manifest={'epoch':'test-retry','deadline':100,'capabilities':{'unit-miner':{}},'checkpoint':{'id':'model'},'max_batches':1}
        class FakeMiner:
            def __init__(self,*args,**kwargs):self.batches=[];events['loads']+=1
            def search(self,*args,**kwargs):
                events['search']+=1
                if events['search']==1:raise RuntimeError('first finite search did not solve')
                self.batches.append(({'env_id':'env','index':0,'checkpoint':'model'},[]))
            def upload(self):events['upload']+=1
        class StopLoop(Exception):pass
        def sleep(_):
            events['sleep']+=1
            if events['sleep']>=2:raise StopLoop()
        with tempfile.TemporaryDirectory() as tmp,patch.object(sys,'argv',['miner','--gateway','https://example.test','--authority','unit','--key','unused','--manifest-url','https://example.test/manifest','--state',tmp]),patch.object(cli,'fetch_signed',return_value=manifest),patch.object(cli,'identity',return_value=SimpleNamespace(id='unit-miner')),patch.object(cli,'checkpoint_download',return_value=Path(tmp)),patch.object(cli,'check_runtime_profile'),patch.object(cli,'entries',return_value=[{'env_id':'env','indices':[0]}]),patch.object(cli,'Miner',FakeMiner),patch.object(cli.time,'time',return_value=0),patch.object(cli.time,'sleep',side_effect=sleep):
            with self.assertRaises(StopLoop):cli.main()
        self.assertEqual(events['search'],2)
        self.assertEqual(events['loads'],1)
        self.assertEqual(events['upload'],1)

    def test_early_freeze_requires_explicit_nonpayable_trial(self):
        from .service import epoch_prefix
        self.assertTrue(epoch_prefix({},True).startswith('nonpayable-'))
        with self.assertRaises(ValueError):epoch_prefix({'epoch_prefix':'live'},False)
        self.assertEqual(epoch_prefix({'epoch_prefix':'live','payable_epochs':True},False),'live')
        with self.assertRaises(ValueError):epoch_prefix({'epoch_prefix':'live','payable_epochs':True},True)

    def test_cached_training_requires_exact_changed_checkpoint_binding(self):
        import json
        from unittest.mock import patch
        from types import SimpleNamespace
        from .service import trained_checkpoint
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);destination=root/'model';destination.mkdir();(root/'epoch-training-metrics.json').write_text(json.dumps({'weights_changed':True,'steps':1,'checkpoint':'trained'}))
            calls=[]
            controller=SimpleNamespace(state=root,publish_checkpoint=lambda path:{'id':'trained'},train=lambda *args:calls.append(args))
            manifest={'epoch':'epoch','checkpoint':{'id':'original'}}
            with patch('subnet.service.model_files',return_value={'model.safetensors':'hash'}):
                result,_=trained_checkpoint(controller,manifest,{},root,destination,1)
                self.assertEqual(result['id'],'trained');self.assertEqual(calls,[])
                controller.publish_checkpoint=lambda path:{'id':'different'}
                with self.assertRaises(ValueError):trained_checkpoint(controller,manifest,{},root,destination,1)



class TensorFramingTests(unittest.TestCase):
    def test_npy_header_limits_before_numpy_allocation(self):
        import io
        import numpy as np
        from unittest.mock import patch
        from .batches import bounded_tensor
        def framed(shape,payload=b'',dtype='<f4'):
            buf=io.BytesIO();np.lib.format.write_array_header_1_0(buf,{'descr':dtype,'fortran_order':False,'shape':shape});return buf.getvalue()+payload
        with patch('subnet.batches.np.load',side_effect=AssertionError('unsafe loader called')):
            for data in (framed((10**12,10**12)),framed((512,200000)),framed((2,3),b'four'),framed((2,3),bytes(25)),framed((1,1),bytes(8),'<f8'),framed((0,2)),framed((1,1),b'x','|O')):
                with self.assertRaises(ValueError):bounded_tensor(data)
        good=np.arange(6,dtype=np.float32).reshape(2,3);buf=io.BytesIO();np.save(buf,good,allow_pickle=False)
        np.testing.assert_array_equal(bounded_tensor(buf.getvalue()),good)
        with self.assertRaises(ValueError):bounded_tensor(buf.getvalue()+b'extra bytes')


class ProofFramingTests(unittest.TestCase):
    def test_malformed_native_proofs_rejected(self):
        import base64
        from .proofs import validate_framing
        valid=base64.b64encode((32769).to_bytes(2,'big')+bytes(256)).decode()
        validate_framing([valid],1)
        for raw in (bytes(258),(1).to_bytes(2,'big')+bytes(256),(65535).to_bytes(2,'big')+bytes(256),bytes(10)):
            with self.assertRaises(ValueError):validate_framing([base64.b64encode(raw).decode()],1)
        with self.assertRaises(ValueError):validate_framing(['not base64!'],1)

if __name__=='__main__':unittest.main()
