import base64, copy, unittest
from nacl.signing import SigningKey
from ops.mining_discovery import project
from subnet.storage import canonical

class DiscoveryControls(unittest.TestCase):
 def setUp(self):
  self.key=SigningKey.generate();self.authority=self.key.verify_key.encode().hex()
  self.config={'source_bundle':{'sha256':'a'*64}}
  self.status={'active':{'epoch':'current','phase':'collect'},'checkpoint':{'id':'pinned','files':{}}}
  self.manifest={'epoch':'current','start':10,'deadline':20,'checkpoint':self.status['checkpoint'],
    'source_bundle':self.config['source_bundle'],'max_batches':3,'audit_policy':{},
    'live_reward_contract':{'version':'live-verified-subset-reward-v1','epoch':'current','payable':True}}
  self.discovery={'authority':self.authority,'expires_at':30,'current_url':'https://test.r2.cloudflarestorage.com/b/current?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=signed'}
 def envelope(self):
  return dict(payload=self.manifest,signer=self.authority,signature=base64.b64encode(self.key.sign(canonical(self.manifest)).signature).decode())
 def check(self,live=True,now=15,envelope=None):
  return project({'chain_weight_submission':True},self.config,self.status,self.discovery,envelope or self.envelope(),self.authority,live=live,now=now)
 def test_original_deadline_and_no_chain_claim(self):
  self.assertTrue(self.check()['accepting_submissions']);self.assertEqual(self.check()['deadline'],20)
  self.assertFalse(self.check(now=20)['accepting_submissions']);self.assertNotIn('upload_capabilities',self.check())
 def test_missing_controller_closes_even_unexpired_window(self):
  self.assertFalse(self.check(live=False)['accepting_submissions'])
 def test_training_and_between_epochs_close(self):
  self.status['active']['phase']='train';self.assertFalse(self.check()['accepting_submissions'])
  self.status['active']=None;self.assertFalse(self.check()['accepting_submissions'])
 def test_wrong_signature_source_checkpoint_and_epoch_refuse(self):
  envelope=self.envelope();envelope['signature']=base64.b64encode(b'x'*64).decode()
  with self.assertRaises(Exception):self.check(envelope=envelope)
  for field,value in [('epoch','old'),('checkpoint',{'id':'other'}),('source_bundle',{'sha256':'b'*64})]:
   original=copy.deepcopy(self.manifest);self.manifest[field]=value
   with self.assertRaises(ValueError):self.check()
   self.manifest=original
 def test_expired_discovery_and_wrong_reward_contract_refuse(self):
  self.discovery['expires_at']=15
  with self.assertRaises(ValueError):self.check()
  self.discovery['expires_at']=30;self.manifest['live_reward_contract']['payable']=False
  with self.assertRaises(ValueError):self.check()
 def test_fresh_read_urls_preserve_identity_but_changed_weights_refuse(self):
  self.manifest['checkpoint']=dict(self.manifest['checkpoint'],read_urls={'model':'fresh-capability'})
  self.assertTrue(self.check()['accepting_submissions'])
  self.manifest['checkpoint']['files']={'model':'different-hash'}
  with self.assertRaises(ValueError):self.check()

if __name__=='__main__':unittest.main()
