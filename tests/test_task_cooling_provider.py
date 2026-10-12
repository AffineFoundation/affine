import base64
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from nacl.signing import SigningKey
from ops import task_cooling_provider as p

class ProviderTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);(self.root/'roles').mkdir();self.key=SigningKey.generate();self.miner=SigningKey.generate();self.auth=self.key.verify_key.encode().hex();p._CACHE.clear();p._SUMMARIES.clear()
  for n in range(1,13):self.epoch(n)
 def tearDown(self):self.tmp.cleanup()
 def sign(self,value,key=None):
  key=key or self.key;return dict(payload=value,signer=key.verify_key.encode().hex(),signature=base64.b64encode(key.sign(p.canonical(value)).signature).decode())
 def write(self,path,value):path.write_bytes(p.canonical(value))
 def epoch(self,n,suffix='original'):
  epoch=f'test--{100+n}-{n}';job_id=epoch+'-train-'+suffix;checkpoint=f'{n:064x}';output=f'{n+1:064x}';source={'subnet/training.py':'a'*64};runtime='c'*64
  manifest=dict(epoch=epoch,checkpoint={'id':checkpoint},source_bundle={'sha256':'b'*64},environments=[dict(env_id='affine_math',indices=[1,2])],K=1,L=1);opening=self.sign(manifest)
  submissions=[];admissions=[];pairs=[]
  for index in(1,2):
   pos=dict(index=index,env_id='affine_math',task_hash=f'{index:064x}',classification='positive',turns=[dict(prompt=[1],output=[3,4])]);neg=dict(pos,classification='negative',turns=[dict(prompt=[1],output=[5,4])]);batch=dict(index=index,env_id='affine_math',epoch=epoch,checkpoint=checkpoint,rollouts=[pos,neg]);bh=p.digest(batch);doc=f'{index+20:064x}';miner=self.miner.verify_key.encode().hex();commit=self.sign(dict(epoch=epoch,checkpoint=checkpoint,miner=miner,batches=[dict(slot=index,index=index,env_id='affine_math',batch_sha256=bh,training_sha256=doc)]),self.miner)
   adm=self.sign(dict(epoch=epoch,checkpoint=checkpoint,document_sha256=doc,document_size=100,original_commitment=commit,miner_identity=miner,slot=index,batch_sha256=bh));submissions.append(dict(learner_admission=adm,sha256=doc,size=100));admissions.append(dict(claimed_batch=batch,batch_sha256=bh,learner_admission_sha256=p.digest(adm),document_sha256=doc,miner_identity=miner));pairs.append(dict(pair_sha256=p.digest(dict(env_id='affine_math',index=index,positive=pos,negative=neg)),task_sha256=p.digest(dict(env_id='affine_math',index=index,task_hash=pos['task_hash']))))
  declaration=self.sign(dict(epoch=epoch,job_id=job_id,original_public_manifest_sha256=p.digest(opening),original_signed_manifest_sha256=p.digest(opening),execution_source_files=source,training_source_bundle={'sha256':'d'*64},training_runtime_sha256=runtime,original_source_bundle_sha256='b'*64,steps=1,optimizer_step_before=n-1))
  job=dict(role='train',job_id=job_id,manifest=opening,source_files=source,unaudited_training_execution=declaration,submissions=submissions,steps=1)
  diag=dict(epoch=epoch,input_checkpoint=checkpoint,global_optimizer_step_before=n-1,global_optimizer_step_after=n,optimizer_steps=1,task_count=2,pair_count=2)
  update=dict(global_optimizer_step=n,epoch_optimizer_step=1,input_checkpoint=checkpoint,pairs=pairs,gradient_pairs=2,unique_verified_pairs=2,gradient_tasks=2,unique_tasks=2)
  report=dict(role='train',job_id=job_id,job_sha256=p.digest(job),epoch=epoch,checkpoint=checkpoint,new_checkpoint={'id':output},success=True,completed_at=90+n,source_files=source,unaudited_training_execution=dict(declaration_sha256=p.digest(declaration),execution_source_files_sha256=p.digest(source),execution_source_bundle_sha256='d'*64),training_runtime_sha256=runtime,training_admissions=admissions,training=dict(global_step_before=n-1,global_step_after=n,steps=1,persistent_diagnostics=diag,updates=[update]))
  self.write(self.root/(epoch+'-first-signed-manifest.json'),opening);self.write(self.root/(epoch+'-signed-learner-completion.json'),self.sign(dict(round=n,epoch=epoch,checkpoint=checkpoint,next_checkpoint=output,completed_at=110+n)));self.write(self.root/'roles'/(job_id+'-job.json'),self.sign(job));self.write(self.root/'roles'/(job_id+'-report.json'),report)
  return epoch,report
 def run_provider(self):return p.evidence_provider({'state':str(self.root)},13,'test--200-13',authority=self.auth)
 def mutate_report(self,n,fn):
  path=next((self.root/'roles').glob(f'*-{n}-train-original-report.json'));value=json.loads(path.read_bytes());fn(value);self.write(path,value)
 def test_complete_exposure_and_fresh_bytes_cache(self):
  a=self.run_provider();self.assertEqual([r['round']for r in a['records']],list(range(1,13)));self.assertEqual(a['records'][0]['updates'],[dict(global_optimizer_step=1,indices=[1,2])]);self.assertEqual(a,self.run_provider());a['records'][0]['updates'][0]['indices'].clear();self.assertEqual(self.run_provider()['records'][0]['updates'][0]['indices'],[1,2])
 def test_failed_original_successful_recovery(self):
  self.mutate_report(4,lambda x:x.update(success=False,new_checkpoint={}));self.epoch(4,'recovery');v=self.run_provider();self.assertIn('-train-recovery-job.json',v['records'][3]['provenance']['job']['path'])
 def test_two_matching_successes_refused(self):
  self.epoch(4,'recovery')
  with self.assertRaisesRegex(p.ExposureUnavailable,'missing_or_ambiguous_successful_recovery'):self.run_provider()
 def test_partial_round_no_closure_refused(self):
  next(self.root.glob('*-12-signed-learner-completion.json')).unlink()
  with self.assertRaisesRegex(p.ExposureUnavailable,'missing_or_ambiguous_closed_round'):self.run_provider()
 def test_corrupt_closure_signature(self):
  path=next(self.root.glob('*-12-signed-learner-completion.json'));v=json.loads(path.read_bytes());v['payload']['next_checkpoint']='f'*64;self.write(path,v)
  with self.assertRaisesRegex(p.ExposureUnavailable,'invalid_signature'):self.run_provider()
 def test_corrupt_original_manifest_signature(self):
  path=next(self.root.glob('*-12-first-signed-manifest.json'));v=json.loads(path.read_bytes());v['payload']['K']=2;self.write(path,v)
  with self.assertRaisesRegex(p.ExposureUnavailable,'invalid_signature'):self.run_provider()
 def test_report_source_mismatch_even_after_cached_hit(self):
  self.run_provider();self.mutate_report(7,lambda x:x['source_files'].update({'subnet/training.py':'f'*64}))
  with self.assertRaisesRegex(p.ExposureUnavailable,'execution_source_binding'):self.run_provider()
 def test_claimed_input_mutation_is_unbound(self):
  self.mutate_report(7,lambda x:x['training_admissions'][0]['claimed_batch']['rollouts'][0]['turns'][0]['output'].append(19))
  with self.assertRaisesRegex(p.ExposureUnavailable,'unbound_reported_batch'):self.run_provider()
 def test_pair_attribution_mutation(self):
  self.mutate_report(7,lambda x:x['training']['updates'][0]['pairs'][0].update(task_sha256='f'*64))
  with self.assertRaisesRegex(p.ExposureUnavailable,'update_pair_attribution'):self.run_provider()
 def test_counter_mismatch(self):
  self.mutate_report(7,lambda x:x['training'].update(global_step_after=1234))
  with self.assertRaisesRegex(p.ExposureUnavailable,'optimizer_counter_binding'):self.run_provider()
 def test_time_budget_falls_back(self):
  with patch.object(p,'MAX_SECONDS',0),self.assertRaisesRegex(p.ExposureUnavailable,'history_time_budget'):self.run_provider()
 def test_symlink_evidence_refused(self):
  path=next(self.root.glob('*-12-first-signed-manifest.json'));target=self.root/'copy.json';path.rename(target);path.symlink_to(target)
  with self.assertRaises(p.ExposureUnavailable):self.run_provider()
 def test_report_size_bound(self):
  with patch.object(p,'MAX_REPORT',1),self.assertRaisesRegex(p.ExposureUnavailable,'evidence_file_bounds'):self.run_provider()
 def test_only_previous_closed_rounds_read(self):
  self.epoch(13);v=self.run_provider();self.assertEqual(v['records'][-1]['round'],12)
 def test_cache_does_not_ignore_added_recovery(self):
  self.run_provider();self.epoch(5,'recovery')
  with self.assertRaisesRegex(p.ExposureUnavailable,'missing_or_ambiguous_successful_recovery'):self.run_provider()
 def test_unsigned_report_limitation_explicit(self):
  v=self.run_provider();self.assertEqual(v['assurance'],'signed-closure-and-job-linked-unsigned-trainer-report');self.assertEqual(set(v['records'][0]['provenance']),{'closure','job','report'})

if __name__=='__main__':unittest.main()
