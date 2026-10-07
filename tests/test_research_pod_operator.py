import base64,copy,json,pathlib,tempfile,time,unittest
from unittest.mock import patch
from nacl.signing import SigningKey
from ops.research_pod_operator import canonical, rental_intent, rent_original, observe_heartbeat, guarded_launch, REGISTRY

class Registry:
 def __init__(self,path):self.path=path;self.rows={};self.fail=False
 def registry_path(self):return self.path
 def load(self):return copy.deepcopy(self.rows)
 def register(self,name,**kw):
  if self.fail:raise OSError('fsync fail')
  self.rows[name]=dict(kw,released_at=None)
 def touch(self,name):self.rows[name]['heartbeat_at']=time.time()

class Operator(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=pathlib.Path(self.temp.name);self.reg=Registry(self.root/'registry.json');self.key=SigningKey.generate();self.authority=self.key.verify_key.encode().hex();now=int(time.time());self.payload=dict(version='retained-research-original-rental-intent-v1',execute_allowed=True,created_at=now,expires_at=now+1200,name='affine-original-test',purpose='test qualification',price_usd_h=5.76,offer_id='exact-offer',template_id='exact-template',gpu_type='H200',gpu_count=1,retention_authorized=True,registry_path=REGISTRY,registry_sha256='a'*64,operator_sha256='b'*64,ownership_module_sha256='c'*64,output_directory=str(self.root/'original'));self.doc=self.sign(self.payload)
 def sign(self,p):return dict(payload=p,signer=self.authority,signature=base64.b64encode(self.key.sign(canonical(p)).signature).decode())
 def provider(self,p):
  self.assertEqual(self.reg.rows[p['name']]['expected_hours'],0);self.assertEqual(self.reg.rows[p['name']]['meta']['provider_pod_id'],'pending-original-rental');self.assertTrue((self.root/'original/original-intent.json').exists());return dict(pod=dict(id='genuine-original-pod',name=p['name'],gpu_type='H200',gpu_count=1,price_per_hour=5.76))
 def test_signed_original_registered_before_provider_and_heartbeat(self):
  b=rent_original(self.reg,self.doc,self.provider,self.root/'original',self.authority);self.assertEqual(b['pod_id'],'genuine-original-pod');actual=json.loads((self.root/'original/original-owned-binding.json').read_bytes());self.assertFalse(actual['node_qualified']);observe_heartbeat(self.reg,self.doc,self.root/'original',self.authority)
 def test_forged_intent_prevents_rental(self):
  forged=copy.deepcopy(self.doc);forged['payload']['offer_id']='other';calls=[]
  with self.assertRaises(Exception):rent_original(self.reg,forged,lambda p:calls.append(1),self.root/'original',self.authority)
  self.assertFalse(calls);self.assertFalse((self.root/'original').exists())
 def test_expired_intent_prevents_new_provider_but_allows_owned_heartbeat(self):
  rent_original(self.reg,self.doc,self.provider,self.root/'original',self.authority)
  with patch('ops.research_pod_operator.time.time',return_value=self.payload['expires_at']+1):
   with self.assertRaises(ValueError):rental_intent(self.doc,self.authority)
   observe_heartbeat(self.reg,self.doc,self.root/'original',self.authority)
 def test_timeout_no_reissue_and_pending_owner_preserved(self):
  calls=[]
  def provider(p):calls.append(1);raise TimeoutError('not a new rental permission')
  with self.assertRaises(TimeoutError):rent_original(self.reg,self.doc,provider,self.root/'original',self.authority)
  with self.assertRaises(ValueError):rent_original(self.reg,self.doc,provider,self.root/'original',self.authority)
  self.assertEqual(calls,[1]);self.assertEqual(self.reg.rows[self.payload['name']]['expected_hours'],0);self.assertTrue((self.root/'original/original-provider-error.json').exists())
 def test_registry_failure_prevents_provider(self):
  self.reg.fail=True;calls=[]
  with self.assertRaises(OSError):rent_original(self.reg,self.doc,lambda p:calls.append(1),self.root/'original',self.authority)
  self.assertFalse(calls)
 def test_guarded_launch_exact_once_and_no_retirement(self):
  b=rent_original(self.reg,self.doc,self.provider,self.root/'original',self.authority);calls=[];journal=self.root/'separate-approved-original-launch.json';guarded_launch(self.reg,b,journal,lambda:calls.append(1))
  with self.assertRaises(FileExistsError):guarded_launch(self.reg,b,journal,lambda:calls.append(1))
  self.assertEqual(calls,[1]);self.assertFalse(self.reg.rows[b['name']].get('release_requested_at'))
 def test_bound_provider_substitution_refuses_heartbeat(self):
  rent_original(self.reg,self.doc,self.provider,self.root/'original',self.authority);p=self.root/'original/original-provider-receipt.json';r=json.loads(p.read_bytes());r['pod']['id']='substituted';p.write_text(json.dumps(r))
  with self.assertRaises(ValueError):observe_heartbeat(self.reg,self.doc,self.root/'original',self.authority)
if __name__=='__main__':unittest.main()
