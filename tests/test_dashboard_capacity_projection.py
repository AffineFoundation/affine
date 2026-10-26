"""Authenticated training capacity and capture-availability projection controls."""
import base64,copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from nacl.signing import SigningKey
from dashboard import learner_projection as p, server as s

class CapacityTests(unittest.TestCase):
 def fixture(self,n=300,selected=None,cap=512):
  selected=n if selected is None else selected
  root=SigningKey.generate();miner=SigningKey.generate();identity=miner.verify_key.encode().hex()
  def signed(payload,key=root):
   return dict(payload=copy.deepcopy(payload),signer=key.verify_key.encode().hex(),signature=base64.b64encode(key.sign(p.canonical(payload)).signature).decode())
  epoch='nonpayable-capacity-fixture--1-117'
  manifest=dict(epoch=epoch,checkpoint={'id':'cp'},source_bundle={'sha256':'source'},start=1,deadline=2,
   training_input_policy='committed-unaudited-training-v1',training_policy='bf16-cpu-fp32-master-task-normalized-persistent-v4',K=4,L=4)
  if cap is not None:manifest['training_task_capacity']=dict(version='signed-training-task-capacity-v1',max_tasks=cap)
  children=[dict(slot=i,batch_sha256=str(i),sha256='proof'+str(i),training_sha256='tokens'+str(i),training_size=100)for i in range(n)]
  commitment=signed(dict(version='small-commitment-pairs-v2',epoch=epoch,checkpoint='cp',source='source',miner=identity,batches=children),miner)
  submissions=[]
  for child in children:
   admission=signed(dict(version='committed-unaudited-training-v1',assurance='unaudited',epoch=epoch,checkpoint='cp',source_sha256='source',miner_identity=identity,slot=child['slot'],commitment_sha256=p.digest(commitment),original_commitment=commitment,batch_sha256=child['batch_sha256'],proof_sha256=child['sha256'],document_sha256=child['training_sha256'],document_size=100))
   submissions.append(dict(learner_admission=admission,sha256=child['training_sha256'],size=100,url='PRIVATE-CAPABILITY'))
  inventory=[dict(learner_admission_sha256=p.digest(row['learner_admission']),sha256=row['sha256'],size=row['size'])for row in submissions]
  selection=dict(version='bounded-postfreeze-learner-selection-v1',eligible_count=n,training_count=selected,unselected_count=n-selected,
   cap=256 if cap is None else cap,eligible_inventory_sha256=p.digest(inventory),selected_inventory_sha256=p.digest(inventory[:selected]))
  population=dict(assurance='unaudited',epoch=epoch,checkpoint='cp',committed_inventory=[dict(miner=identity,commitment_document=commitment,commitment_sha256=p.digest(commitment))],committed_count=n,eligible_count=n,training_count=selected,eligible_inventory=inventory,training_selection=selection)
  document=dict(version='committed-unaudited-training-v1',population=population,submissions=submissions[:selected])
  return document,manifest,{identity:85},root.verify_key.encode().hex(),signed

 def test_signed512_accepts_more_than256_with_exact_counts(self):
  d,m,u,a,sign=self.fixture();out=p.project(d,m,u,a,manifest_envelope=sign(m))
  self.assertEqual((out['submitted'],out['learner_training_selected'],out['submitting_identities']),(300,300,1))
  self.assertEqual(sum(out['submitted_grid']),300)
  self.assertFalse(out['proof_verification_claimed']);self.assertNotIn('PRIVATE',json.dumps(out))

 def test_historical256_stays_supported_without_manifest_envelope(self):
  d,m,u,a,_=self.fixture(selected=256,cap=None);out=p.project(d,m,u,a)
  self.assertEqual((out['submitted'],out['learner_training_selected']),(300,256))

 def test_signed_capacity_follows_explicit256_and1024(self):
  for cap,selected in ((256,256),(1024,300)):
   d,m,u,a,sign=self.fixture(n=300,selected=selected,cap=cap)
   m.update(K=8,L=8,training_policy='future-root-authorized-training-policy')
   out=p.project(d,m,u,a,manifest_envelope=sign(m))
   self.assertIsNotNone(out)
   self.assertEqual(out['learner_training_selected'],selected)

 def test_historical_over256_selected_is_rejected(self):
  d,m,u,a,_=self.fixture(selected=257,cap=None);self.assertIsNone(p.project(d,m,u,a))

 def test_explicit512_requires_authenticated_manifest(self):
  d,m,u,a,sign=self.fixture()
  self.assertIsNone(p.project(d,m,u,a))
  envelope=sign(m);envelope['signature']=base64.b64encode(b'x'*64).decode()
  self.assertIsNone(p.project(d,m,u,a,manifest_envelope=envelope))

 def test_bad_capacity_schema_and_caps_rejected_even_with_valid_signature(self):
  d,m,u,a,sign=self.fixture(n=2)
  for capacity in ({'version':'signed-training-task-capacity-v1','max_tasks':True},
   {'version':'signed-training-task-capacity-v1','max_tasks':0},
   {'version':'signed-training-task-capacity-v1','max_tasks':-1},
   {'version':'signed-training-task-capacity-v1','max_tasks':512.0},
   {'version':'signed-training-task-capacity-v1','max_tasks':'512'},
   {'version':'unknown','max_tasks':512},
   {'version':'signed-training-task-capacity-v1','max_tasks':512,'extra':1},
   {'max_tasks':512},None):
   bad=copy.deepcopy(m);bad['training_task_capacity']=capacity
   with self.subTest(capacity=capacity):self.assertIsNone(p.project(d,bad,u,a,manifest_envelope=sign(bad)))

 def test_signed_opening_must_match_scope_and_contract(self):
  d,m,u,a,sign=self.fixture(n=2)
  for name,value in (('epoch','other'),('source_bundle',{'sha256':'other'}),('checkpoint',{'id':'other'}),('K',2),('L',True),('training_policy','other')):
   bad=copy.deepcopy(m);bad[name]=value
   with self.subTest(name=name):self.assertIsNone(p.project(d,m,u,a,manifest_envelope=sign(bad)))
  bad=copy.deepcopy(m);bad['training_input_policy']='other'
  self.assertIsNone(p.project(d,bad,u,a,manifest_envelope=sign(bad)))

 def test_signed_scope_binding_does_not_equate_bool_and_integer(self):
  d,m,u,a,sign=self.fixture(n=2);m['K']=1
  bad=copy.deepcopy(m);bad['K']=True
  self.assertIsNone(p.project(d,m,u,a,manifest_envelope=sign(bad)))

 def test_selection_cap_must_match_signed_capacity(self):
  d,m,u,a,sign=self.fixture(n=2);d['population']['training_selection']['cap']=256
  self.assertIsNone(p.project(d,m,u,a,manifest_envelope=sign(m)))

 def test_selected_count_cannot_exceed_signed_capacity(self):
  d,m,u,a,sign=self.fixture(n=33,cap=32)
  self.assertIsNone(p.project(d,m,u,a,manifest_envelope=sign(m)))

 def test_legacy_selected331_never_becomes_representative512(self):
  d,m,u,a,sign=self.fixture(n=400,selected=331)
  out=p.project(d,m,u,a,manifest_envelope=sign(m))
  self.assertEqual(out['learner_training_selected'],331)
  self.assertEqual(out['submitted'],400)

 def database_row(self,state):
  d,m,u,a,sign=self.fixture(n=270)
  with tempfile.TemporaryDirectory(prefix='dashboard-capacity-control-')as temp:
   root=Path(temp);source=root/'state';folder=source/'live-math-launch-preparation-v1/distributed-preparation/live-controller-v1/controller-state';folder.mkdir(parents=True)
   epoch=m['epoch'];(folder/f'{epoch}-manifest.json').write_text(json.dumps(m))
   (folder/f'{epoch}-first-signed-manifest.json').write_text(json.dumps(sign(m)))
   (folder/f'{epoch}-registrations.json').write_text(json.dumps({'miner':dict(public_key=next(iter(u)),uid=85)}))
   if state=='valid':(folder/f'{epoch}-learner-population.json').write_text(json.dumps(d))
   elif state=='invalid':(folder/f'{epoch}-learner-population.json').write_text('{malformed')
   elif state=='tampered':
    d['population']['eligible_count']+=1;(folder/f'{epoch}-learner-population.json').write_text(json.dumps(d))
   with patch.object(s,'project_learner',side_effect=lambda d,m,u,**kw:p.project(d,m,u,a,**kw)):
    db=s.Database(root/'fixture.sqlite',source);db.refresh();return db.snapshot()['epochs'][0]

 def test_server_passes_signed_capacity_and_restores_captured_counts(self):
  row=self.database_row('valid')
  self.assertEqual((row['batches'],row['submissions'],row['unchecked']),(270,1,270))
  self.assertTrue(row['batches_available']);self.assertNotIn('batch_count_status',row)

 def test_missing_capture_remains_awaiting_capture(self):
  row=self.database_row('missing');self.assertEqual(row['batch_count_status'],'awaiting_capture');self.assertFalse(row['batches_available'])

 def test_invalid_present_capture_is_not_reported_as_missing(self):
  for state in ('invalid','tampered'):
   row=self.database_row(state)
   self.assertEqual(row['batch_count_status'],'invalid_capture_projection')
   self.assertEqual(row['batch_count_source'],'unavailable-invalid-capture-projection')
   self.assertFalse(row['batches_available'])

if __name__=='__main__':unittest.main()
