import base64,copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from nacl.signing import SigningKey
from dashboard.learner_projection import canonical,digest,project
from dashboard.server import Database

class LearnerProjectionTests(unittest.TestCase):
 def fixture(self,n,eligible,epoch='test-15'):
  root=SigningKey.generate();miner=SigningKey.generate();identity=miner.verify_key.encode().hex()
  def signed(p,key):return dict(payload=p,signer=key.verify_key.encode().hex(),signature=base64.b64encode(key.sign(canonical(p)).signature).decode())
  manifest=dict(epoch=epoch,checkpoint={'id':'cp'},source_bundle={'sha256':'source'},start=15,deadline=16)
  children=[dict(slot=i,batch_sha256=str(i),sha256='proof'+str(i),training_sha256='tokens'+str(i),training_size=100)for i in range(n)]
  commitment=signed(dict(version='small-commitment-pairs-v2',epoch=epoch,checkpoint='cp',source='source',miner=identity,batches=children),miner)
  submissions=[]
  for c in children[:eligible]:
   admission=signed(dict(version='committed-unaudited-training-v1',assurance='unaudited',epoch=epoch,checkpoint='cp',source_sha256='source',miner_identity=identity,slot=c['slot'],commitment_sha256=digest(commitment),original_commitment=commitment,batch_sha256=c['batch_sha256'],proof_sha256=c['sha256'],document_sha256=c['training_sha256'],document_size=100),root)
   submissions.append(dict(learner_admission=admission,sha256=c['training_sha256'],size=100,url='PRIVATE-CAPABILITY'))
  population=dict(assurance='unaudited',epoch=epoch,checkpoint='cp',committed_inventory=[dict(miner=identity,commitment_document=commitment,commitment_sha256=digest(commitment))],committed_count=n,eligible_count=eligible,eligible_inventory=[dict(learner_admission_sha256=digest(s['learner_admission']),sha256=s['sha256'],size=s['size'])for s in submissions])
  return dict(version='committed-unaudited-training-v1',population=population,submissions=submissions),manifest,{identity:85},root.verify_key.encode().hex()
 def test_signed_population_is_unaudited_counts_not_accepted(self):
  d,m,u,a=self.fixture(270,240);r=project(d,m,u,a)
  self.assertEqual((r['submitted'],r['learner_eligible'],r['learner_excluded']),(270,240,30));self.assertEqual(r['submitted_grid'][85],270)
  self.assertFalse(r['proof_verification_claimed']);self.assertNotIn('PRIVATE',json.dumps(r))
 def test_tampered_replayed_and_duplicate_inputs_unavailable(self):
  d,m,u,a=self.fixture(3,2)
  for mutate in [lambda x:x['population'].__setitem__('eligible_count',3),lambda x:x['submissions'].append(x['submissions'][0]),lambda x:x['submissions'][0]['learner_admission']['payload'].__setitem__('checkpoint','old'),lambda x:x['population']['committed_inventory'][0]['commitment_document']['payload'].__setitem__('epoch','old')]:
   bad=copy.deepcopy(d);mutate(bad);self.assertIsNone(project(bad,m,u,a))
 def test_bounded_training_subset_preserves_full_eligible_population(self):
  d,m,u,a=self.fixture(359,309)
  full=d['population']['eligible_inventory'];d['submissions']=d['submissions'][:256]
  selected=[dict(learner_admission_sha256=digest(x['learner_admission']),sha256=x['sha256'],size=x['size'])for x in d['submissions']]
  d['population'].update(training_count=256,training_selection=dict(version='bounded-postfreeze-learner-selection-v1',eligible_count=309,training_count=256,unselected_count=53,cap=256,eligible_inventory_sha256=digest(full),selected_inventory_sha256=digest(selected)))
  r=project(d,m,u,a)
  self.assertEqual((r['submitted'],r['learner_eligible'],r['learner_training_selected'],r['learner_excluded']),(359,309,256,50))
  self.assertEqual(r['eligible_grid'][85],309);self.assertFalse(r['proof_verification_claimed']);self.assertNotIn('PRIVATE',json.dumps(r))
  for mutate in [lambda x:x['population']['eligible_inventory'][-1].__setitem__('sha256','fake'),lambda x:x['population']['eligible_inventory'][-1].__setitem__('size',999),lambda x:x['population']['eligible_inventory'].append(x['population']['eligible_inventory'][0]),lambda x:x['population']['training_selection'].__setitem__('training_count',309),lambda x:x['population']['eligible_inventory'][0].__setitem__('learner_admission_sha256','0'*64),lambda x:x['submissions'].append(x['submissions'][0])]:
   bad=copy.deepcopy(d);mutate(bad);self.assertIsNone(project(bad,m,u,a))
 def test_database_legacy_and_active_learner_counts_no_double_count(self):
  with tempfile.TemporaryDirectory()as tmp:
   source=Path(tmp)/'state';folder=source/'live';folder.mkdir(parents=True)
   (folder/'test-13-manifest.json').write_text(json.dumps(dict(epoch='test-13',start=13,deadline=14)))
   (folder/'test-13-verified.json').write_text(json.dumps({'legacy':dict(accepted=[{}]*10,outcomes=[dict(valid=True)]*10)}))
   # Same trusted operator authority across the two populations.
   d14,m14,u14,a=self.fixture(165,161,'test-14');d15,m15,u15,_=self.fixture(270,240,'test-15')
   from dashboard import server
   for d,m,u in [(d14,m14,u14),(d15,m15,u15)]:
    eid=m['epoch'];(folder/f'{eid}-manifest.json').write_text(json.dumps(m));(folder/f'{eid}-learner-population.json').write_text(json.dumps(d));(folder/f'{eid}-registrations.json').write_text(json.dumps({'hotkey':dict(public_key=next(iter(u)),uid=85)}))
   (folder/'controller.json').write_text(json.dumps(dict(active=dict(epoch='test-15',phase='train'))))
   authorities={'test-14':a,'test-15':d15['submissions'][0]['learner_admission']['signer']}
   with patch.object(server,'project_learner',side_effect=lambda d,m,u:project(d,m,u,authorities.get(m['epoch'],a))):
    db=Database(Path(tmp)/'db',source);db.refresh();rows={r['id']:r for r in db.snapshot()['epochs']}
   self.assertEqual(rows['test-13']['accepted'],10)
   self.assertEqual((rows['test-14']['batches'],rows['test-14']['learner_eligible']),(165,161))
   self.assertEqual((rows['test-15']['batches'],rows['test-15']['learner_eligible'],rows['test-15']['accepted'],rows['test-15']['unchecked']),(270,240,0,270))
   self.assertFalse(rows['test-15']['audit_breakdown_available']);self.assertEqual(rows['test-15']['phase'],'train');self.assertTrue(rows['test-15']['batches_available'])
