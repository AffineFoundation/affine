import base64,copy,unittest
from nacl.signing import SigningKey
from subnet.remote_backend import publication_request,PUBLICATION_PROJECTION_VERSION
from subnet.storage import canonical
class FutureProjectionTests(unittest.TestCase):
 def setUp(self):
  self.k=SigningKey.generate();self.authority=self.k.verify_key.encode().hex();self.policy={'version':PUBLICATION_PROJECTION_VERSION}
  self.m={'epoch':'nonpayable-e21','checkpoint':{'id':'b'*64},'source_bundle':{'sha256':'a'*64}}
 def sign(self,d):return dict(payload=d,signer=self.authority,signature=base64.b64encode(self.k.sign(canonical(d)).signature).decode())
 def recovery(self):return self.sign(dict(version='terminal-parent-restore-pre-update-bootstrap-recovery-v3',epoch=self.m['epoch'],replacement_execution_source_sha256='a'*64))
 def test_default_off_original_object_and_label(self):
  self.m['training_startup_recovery']=self.recovery();label,m=publication_request(self.m,self.authority)
  self.assertIs(m,self.m);self.assertEqual(label,'nonpayable-e21-publish-bbbbbbbb')
 def test_normal_upload_policy_preserves_original_object(self):
  self.assertIs(publication_request(self.m,self.authority,self.policy)[1],self.m)
 def test_authenticated_projection_only_field_fresh_label(self):
  self.m['training_startup_recovery']=self.recovery();before=copy.deepcopy(self.m);label,m=publication_request(self.m,self.authority,self.policy)
  self.assertEqual(self.m,before);self.assertEqual(set(self.m)-set(m),{'training_startup_recovery'});self.assertTrue(label.endswith('-publication-v1'));self.assertEqual(m['checkpoint'],self.m['checkpoint'])
 def test_foreign_signature_null_scope_source_version_rejected(self):
  for change in ('signature','null','epoch','source','version'):
   with self.subTest(change=change):
    d=self.recovery()
    if change=='signature':d['signature']='invalid'
    elif change=='null':d=None
    else:
     v=d['payload'];v[{'epoch':'epoch','source':'replacement_execution_source_sha256','version':'version'}[change]]='foreign';d=self.sign(v)
    m=dict(self.m,training_startup_recovery=d)
    with self.assertRaises(Exception):publication_request(m,self.authority,self.policy)
 def test_unknown_policy_refused(self):
  with self.assertRaises(ValueError):publication_request(self.m,self.authority,{'version':'foreign'})
