"""Prospective durable gating with real signed CPU optimizer publications."""
import copy,json,tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import test_persistent_training_integration as fixtures
from subnet.reward_publication import VERSION,emit,require,PublicationPending
from subnet.persistent_training_protocol import independently_commit
from subnet.remote_backend import save
from subnet.storage import canonical
from subnet.live_reward_bridge import sha
class RewardPublicationTests(unittest.TestCase):
 def setUp(self):
  self.f=fixtures.PersistentIntegrationTests();self.f.setUp();self.addCleanup(self.f.doCleanups);f=self.f;f.manifest['reward_publication_policy']=VERSION
  from training_receipt_fixtures import signed_receipt
  receipt,_,_,_=signed_receipt(f.key,f.manifest,f.miner,f.receipts[f.miner],f.batch)
  f.submission['verifier_receipt']=receipt
  self.epoch=f.manifest['epoch'];self.score=dict(epoch_id=self.epoch,points={f.miner:1},receipts=f.receipts,finalized_at=25)
  save(f.root/(self.epoch+'-scores.json'),self.score);save(f.root/(self.epoch+'-signed-compute-scores.json'),f.sign(self.score))
 def prepared(self):
  f=self.f;report,job=f.report();pointer=independently_commit(f.controller,report,job,f.manifest)
  cp=dict(f.cp,descriptor_key='public/checkpoint-authority.json');f.bucket.json(cp['descriptor_key'],f.sign(dict(id=cp['id'],files=cp['files'])))
  save(f.root/'roles'/(self.epoch+'-train.json'),dict(job_id=job['job_id'],job_sha256=sha(job)))
  save(f.root/'roles'/(job['job_id']+'-job.json'),f.sign(job));save(f.root/'roles'/(job['job_id']+'-report.json'),report)
  save(f.root/'latest-trainer-state.json',pointer)
  receipt=dict(checkpoint=cp['id'],operator_independent_hashes=True,objects={n:dict(sha256=h)for n,h in cp['files'].items()});save(f.root/(self.epoch+'-checkpoint-publication.json'),receipt)
  metrics=dict(trainer_state=pointer,steps=3,source_epoch=self.epoch,input_checkpoint=f.cp['id'],new_checkpoint=cp,original_job_sha256=sha(job),trainer_binding_sha256=sha(f.manifest['trainer_state_binding']));save(f.root/(self.epoch+'-training-metrics.json'),metrics)
  return report,job,metrics
 def test_signed_scores_alone_not_chain_eligible(self):
  with self.assertRaises(PublicationPending):require(self.f.root,self.f.manifest,self.f.authority)
  with self.assertRaises(PublicationPending):emit(self.f.controller,self.f.manifest)
 def test_real_cpu_optimizer_and_checkpoint_authority_release_gate(self):
  self.prepared();ready=emit(self.f.controller,self.f.manifest);self.assertEqual(ready['evidence']['optimizer_steps'],3);self.assertEqual(require(self.f.root,self.f.manifest,self.f.authority),ready)
  # Later actual trainer journal does not invalidate immutable old reward history.
  save(self.f.root/'latest-trainer-state.json',dict(later=True));self.assertEqual(require(self.f.root,self.f.manifest,self.f.authority),ready)
 def test_missing_either_durable_publication_never_emits(self):
  _,_,m=self.prepared();f=self.f
  for key in [m['trainer_state']['descriptor_key'],m['new_checkpoint']['descriptor_key']]:
   body=f.bucket.objects.pop(key)
   with self.assertRaises(Exception):emit(f.controller,f.manifest)
   self.assertFalse((f.root/(self.epoch+'-reward-publication-ready.json')).exists());f.bucket.objects[key]=body
 def test_tampered_signed_evidence_metrics_and_scores_fail(self):
  _,_,m=self.prepared();f=self.f;emit(f.controller,f.manifest)
  for label,value in [('training-metrics',dict(m,steps=True)),('scores',dict(self.score,points={f.miner:9}))]:
   p=f.root/(self.epoch+'-'+label+'.json');old=p.read_bytes();save(p,value)
   with self.assertRaises(ValueError):require(f.root,f.manifest,f.authority)
   p.write_bytes(old)
  p=f.root/(self.epoch+'-reward-publication-ready.json');d=json.loads(p.read_text());d['payload']['evidence']['output_checkpoint']='f'*64;save(p,d)
  with self.assertRaises(Exception):require(f.root,f.manifest,f.authority)
 def test_zero_points_needs_explicit_no_update_close_not_fake_training(self):
  f=self.f;self.score['points']={f.miner:0};save(f.root/(self.epoch+'-scores.json'),self.score);save(f.root/(self.epoch+'-signed-compute-scores.json'),f.sign(self.score))
  with self.assertRaises(PublicationPending):emit(f.controller,f.manifest)
  save(f.root/(self.epoch+'-empty-closed.json'),dict(epoch=self.epoch,status='closed_no_accepted_batches',payable=False,checkpoint=f.cp['id']))
  ready=emit(f.controller,f.manifest);self.assertEqual(ready['evidence']['training_steps'],0);self.assertEqual(ready['evidence']['status'],'closed_no_update')
 def test_public_put_local_crash_recovers_exact_original_attestation(self):
  self.prepared();f=self.f
  with patch('subnet.remote_backend.save',side_effect=OSError('actual simulated local interruption')):
   with self.assertRaises(OSError):emit(f.controller,f.manifest)
  key='public/'+self.epoch+'/reward-publication-ready.json';original=f.bucket.objects[key]
  ready=emit(f.controller,f.manifest);self.assertEqual(f.bucket.objects[key],original)
  self.assertEqual(ready,json.loads(original)['payload'])
 def test_writer_and_direct_exporter_block_new_policy_before_training(self):
  from test_live_reward_completeness import population
  from ops import live_reward_writer as writer
  from ops import live_reward_exporter as exporter
  from test_live_reward_bridge import sign
  with tempfile.TemporaryDirectory()as tmp:
   key,i,regs,reports,state,reward,epoch,put=population(tmp)
   manifest=dict(i['manifest_document']['payload'],reward_publication_policy=VERSION)
   put('manifest',manifest);put('first-signed-manifest',sign(manifest,key));put('signed-compute-scores',i['score_document'])
   for miner,doc in i['audit_documents'].items():put('signed-compute-audit-'+miner,doc)
   c=dict(compute_state=str(state),reward_state=str(reward))
   with self.assertRaises(writer.FinalizationPending):writer.finalized_reward_completeness(c,i['anchor_document'],i['authority'],window_end=7200)
   with self.assertRaises(PublicationPending):exporter.run_once(state,reward,i['anchor_document'],i['authority'],key,regs,7200)
   self.assertFalse((reward/'hour-7200-reward-units.json').exists())
 def test_legacy_manifests_and_unknown_policy(self):
  m=copy.deepcopy(self.f.manifest);m.pop('reward_publication_policy');self.assertIsNone(require(self.f.root,m,self.f.authority));self.assertIsNone(emit(self.f.controller,m))
  m['reward_publication_policy']='silently-skip-training'
  with self.assertRaises(ValueError):require(self.f.root,m,self.f.authority)
if __name__=='__main__':unittest.main()
