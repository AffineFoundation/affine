"""Real base/frozen Controller.open CPU calls; no model or network."""
import base64,copy,hashlib,importlib.util,json,os,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
from nacl.signing import SigningKey
from subnet.remote_backend import RemoteController
from subnet.controller import Controller
from subnet.storage import canonical
from subnet.learner_blacklist_selection import FIELD,admit

class Opening(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.state=Path(self.tmp.name);self.key=SigningKey.generate();self.auth=self.key.verify_key.encode().hex();self.cp={'id':'d'*64,'files':{}};self.source={'sha256':'e'*64};self.epoch='nonpayable-original-open'
  self.c=RemoteController.__new__(RemoteController);self.c.state=self.state;self.c.authority=SimpleNamespace(id=self.auth);self.c.signed=self.sign;self.c.gateway=SimpleNamespace(open=Mock(return_value={'a'*64:None}),direct_r2=False,epochs={self.epoch:{'start':3700}});self.c.bucket=SimpleNamespace(json=Mock());self.c.independent_state_reader=None
  (self.state/'controller.json').write_bytes(canonical(dict(round=40,active={'epoch':self.epoch,'phase':'opening'})))
  self.audit=dict(version='continuous-probabilistic-audit-v3',recent_epochs=8,decay=.8,prior_alpha=1,prior_beta=1,invalid_multiplier=.1,zero_epoch_after=2,blacklist_after=3,blacklist_epochs=4)
  assessment=self.sign(dict(version='hourly-current-miner-assessment-v1',cutoff=3600,evidence_cutoff=3600,assessment_stale=False,writer_policy_sha256='f'*64,miner_estimates={}))
  self.policy=self.sign(dict(version='confirmed-blacklist-training-selection-v1',checkpoint=self.cp['id'],source_sha256=self.source['sha256'],target_round=40,maximum_age_seconds=3600,assessment_document=assessment,writer_policy_sha256='f'*64,audit_policy=self.audit))
 def sign(self,p):return dict(payload=p,signer=self.auth,signature=base64.b64encode(self.key.sign(canonical(p)).signature).decode())
 def call(self,**changes):
  kw=dict(source_bundle=self.source,max_batches=3,**{FIELD:self.policy,'learner_blacklist_selection_round':40});kw.update(changes)
  with patch('time.time',return_value=3700):return self.c.open(self.epoch,self.cp,['a'*64],**kw)
 def test_actual_base_signature_rejects_old_route(self):
  with self.assertRaises(TypeError):Controller.open(self.c,self.epoch,self.cp,[],**{FIELD:self.policy,'learner_blacklist_selection_round':40})
  self.c.gateway.open.assert_not_called();self.c.bucket.json.assert_not_called()
 def assert_success(self,base_module=None):
  first=[]
  import subnet.controller as module
  module=base_module or module
  original_save=module.save_manifest
  def save(p,m):first.append(copy.deepcopy(m));return original_save(p,m)
  with patch.object(module,'save_manifest',save):m=self.call()
  self.assertTrue(first);self.assertEqual(first[0][FIELD],self.policy);self.assertEqual(first[0]['learner_blacklist_selection_round'],40);self.assertEqual(first[0]['max_batches'],3)
  path=self.state/(self.epoch+'-manifest.json');self.assertEqual(json.loads(path.read_bytes()),m)
  published=[v[0][1]for v in self.c.bucket.json.call_args_list if v[0][0].endswith('/manifest.json')]
  self.assertEqual(len(published),1);self.assertEqual(published[0]['payload'],m)
  self.key.verify_key.verify(canonical(m),base64.b64decode(published[0]['signature']));self.assertEqual(m[FIELD],self.policy)
  self.assertEqual(set(m['capabilities']),{'a'*64});self.assertEqual((m['K'],m['L']),(1,1));return m
 def test_first_local_and_first_public_manifest_complete(self):self.assert_success()
 def test_invalid_policy_refuses_before_gateway_save_and_publication(self):
  for field,value in [('checkpoint','9'*64),('source_sha256','9'*64),('target_round',39)]:
   p=copy.deepcopy(self.policy['payload']);p[field]=value
   with self.subTest(field=field),self.assertRaises(ValueError):self.call(**{FIELD:self.sign(p)})
  for changes in [{FIELD:None},{'learner_blacklist_selection_round':True},{'learner_blacklist_selection_round':39}]:
   with self.assertRaises(ValueError):self.call(**changes)
  self.c.gateway.open.assert_not_called();self.c.bucket.json.assert_not_called();self.assertFalse((self.state/(self.epoch+'-manifest.json')).exists())
 def test_missing_pair_and_wrong_saved_epoch_refuse(self):
  with self.assertRaises(ValueError):self.c.open(self.epoch,self.cp,[],source_bundle=self.source,**{FIELD:self.policy})
  with self.assertRaises(ValueError):self.c.open(self.epoch,self.cp,[],source_bundle=self.source,learner_blacklist_selection_round=40)
  (self.state/'controller.json').write_bytes(canonical(dict(round=40,active={'epoch':'different','phase':'opening'})))
  with self.assertRaises(ValueError):self.call()
  self.c.gateway.open.assert_not_called()
 def test_existing_collect_train_phase_refuses_before_side_effects(self):
  for phase in ('mine','collect','train','after'):
   (self.state/'controller.json').write_bytes(canonical(dict(round=40,active={'epoch':self.epoch,'phase':phase})))
   with self.subTest(phase=phase),self.assertRaises(ValueError):self.call()
  self.c.gateway.open.assert_not_called();self.c.bucket.json.assert_not_called()
 def test_default_off_does_not_read_controller_or_new_metadata(self):
  (self.state/'controller.json').unlink()
  import builtins
  original_import=builtins.__import__
  def no_selection_import(name,*args,**kwargs):
   if 'learner_blacklist_selection'in name:raise AssertionError('default-off new helper dependency')
   return original_import(name,*args,**kwargs)
  with patch('time.time',return_value=3700),patch('builtins.__import__',side_effect=no_selection_import):m=self.c.open(self.epoch,self.cp,[],source_bundle=self.source,max_batches=3)
  self.assertNotIn(FIELD,m);self.assertNotIn('learner_blacklist_selection_round',m)
 def test_saved_opening_keeps_exact_policy_no_refresh(self):
  m=self.assert_success();path=self.state/(self.epoch+'-manifest.json');raw=path.read_bytes()
  # Exercise exact saved-opening branch from gpu_service.run, not a separate
  # implementation: it must never call controller.open/prepare a new policy.
  import ast
  from subnet import gpu_service
  tree=ast.parse(Path(gpu_service.__file__).read_bytes());branches=[n for n in ast.walk(tree)if isinstance(n,ast.If)and ast.unparse(n.test)=='manifestpath.exists()'and any(isinstance(x,ast.Assign)for x in n.body)]
  self.assertEqual(len(branches),1);node=branches[0];self.assertEqual(len(node.body),1)
  scope=dict(manifestpath=path,json=json);exec(compile(ast.Module(body=node.body,type_ignores=[]),'saved-original-opening','exec'),scope)
  self.assertEqual(scope['manifest'],m);self.assertEqual(path.read_bytes(),raw);self.assertEqual(scope['manifest'][FIELD],self.policy)

class FrozenOpening(Opening):
 def test_actual_frozen_base_open(self):
  root=os.environ.get('AFFINE_BLACKLIST_FROZEN_SOURCE')
  if not root:self.skipTest('optional already-admitted frozen dependency route')
  p=Path(root)/'subnet/controller.py'
  # Exact hash is read from the public f213 inventory fixture; no network.
  fixture=Path(__file__).parent/'fixtures/cpu_selection_peer/f213-runtime177.json';expected=json.loads(fixture.read_bytes())['subnet/controller.py'];self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),expected)
  spec=importlib.util.spec_from_file_location('subnet._test_frozen_controller',p);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
  with patch.object(Controller,'open',mod.Controller.open):self.assert_success(mod)
if __name__=='__main__':unittest.main()
