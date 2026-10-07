import copy,importlib.util,json,pathlib,sys,tempfile,types,unittest,base64
from nacl.signing import SigningKey
from ops import durable_learner_service as m,durable_audit_services as g
class Boundary(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=pathlib.Path(self.tmp.name);self.key=SigningKey.generate();self.authority=self.key.verify_key.encode().hex();self.cfg={'state':str(self.root)};self.state=dict(round=39,checkpoint={'id':'a'*64},active=dict(epoch='original-39',phase='mine'))
  self.write();payload=dict(epoch='original-39',checkpoint=self.state['checkpoint'],source_bundle={'sha256':'f'*64});d=dict(payload=payload,signer=self.authority,signature=base64.b64encode(self.key.sign(g.canonical(payload)).signature).decode());opening=self.root/'first-opening.json';opening.write_bytes(g.canonical(d));opening.chmod(0o600)
  self.p=dict(source_sha256='f'*64,stop_after_current_round=dict(version='signed-current-round-CPU-stop-v1',round=39,epoch='original-39',original_opening=dict(path=str(opening),file_sha256=g.file_hash(opening),payload_sha256=g.digest(payload))))
 def write(self):
  p=self.root/'controller.json';p.write_bytes(g.canonical(self.state));p.chmod(0o600)
 def test_same_original_resumes_once_then_natural_boundary(self):
  calls=[]
  def run(cfg,once=False):
   calls.append(once);self.state.update(round=40,active=None);self.write()
  m.run_original_boundary(types.SimpleNamespace(run=run),self.p,self.cfg,self.authority);self.assertEqual(calls,[True])
 def test_defaultoff_calls_original_unbounded_method(self):
  calls=[];m.run_original_boundary(types.SimpleNamespace(run=lambda cfg:calls.append('original')),{},self.cfg,self.authority);self.assertEqual(calls,['original'])
 def test_unpublished_opening_refuses_before_any_execution(self):
  self.state['active']['phase']='opening';self.write();calls=[]
  with self.assertRaises(ValueError):m.run_original_boundary(types.SimpleNamespace(run=lambda *a,**k:calls.append(True)),self.p,self.cfg,self.authority)
  self.assertFalse(calls)
 def test_passed_boundary_or_different_saved_epoch_refuses(self):
  for round,epoch in [(40,'original-40'),(39,'different-39')]:
   self.state.update(round=round,active={'epoch':epoch,'phase':'mine'});self.write()
   with self.assertRaises(ValueError):m.validate_stop_boundary(self.p,self.cfg,self.authority)
 def test_round_bool_and_null_refuse(self):
  bad=copy.deepcopy(self.p);bad['stop_after_current_round']['round']=True
  with self.assertRaises(ValueError):m.validate_stop_boundary(bad,self.cfg,self.authority)
  with self.assertRaises(ValueError):m.validate_stop_boundary({'stop_after_current_round':None},self.cfg,self.authority)
 def test_not_closed_or_skipped_two_rounds_refuses(self):
  def run(cfg,once=False):self.state.update(round=41,active=None);self.write()
  with self.assertRaisesRegex(ValueError,'natural'):m.run_original_boundary(types.SimpleNamespace(run=run),self.p,self.cfg,self.authority)
 def test_source_opening_and_signature_tamper_refuse(self):
  bad=copy.deepcopy(self.p);bad['source_sha256']='b'*64
  with self.assertRaises(ValueError):m.validate_stop_boundary(bad,self.cfg,self.authority)
if __name__=='__main__':unittest.main()
