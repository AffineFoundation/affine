import base64,copy,hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from nacl.signing import SigningKey
from dashboard.cached_evaluator_projection import rows,POLICY
from dashboard.learner_projection import canonical
class ProjectionTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.state=self.root/'eval';self.prod=self.root/'production';self.prod.mkdir();self.key=SigningKey.generate();self.auth=self.key.verify_key.encode().hex()
  for folder in ('checkpoint-evaluations','roles','cache-disposal','durable-evaluation-acks'):(self.state/folder).mkdir(parents=True)
  values=[dict(native_graded=True,proof_verification_performed=False,reward=int(i<19),task_hash=str(i),seed=20261002+i*1000)for i in range(32)]
  manifest=dict(checkpoint={'id':'cp10'},source_bundle={'sha256':'source'},epoch='diagnostic-cp10')
  job=dict(heldout=[dict(indices=list(range(32)),seeds=[20261002+i*1000 for i in range(32)],harness={'max_output_tokens':1024})],role='evaluate',owned_evaluation_policy=POLICY,manifest=self.sign(manifest))
  report=dict(heldout=values,heldout_failures=[])
  ack=dict(original_job=self.sign(job),original_report=report,durable_report_full_readback=True,report_sha256=hashlib.sha256(canonical(report)).hexdigest())
  raw=canonical(self.sign(ack));(self.state/'durable-evaluation-acks/jid.json').write_bytes(raw)
  (self.state/'cache-disposal/jid.json').write_text(json.dumps(dict(result={'status':'complete'},durable_ack_sha256=hashlib.sha256(raw).hexdigest())))
  (self.state/'roles/label.json').write_text('{"job_id":"jid"}')
  self.record=dict(timestamp=2,seed=20261002,experiment_id='owned-cached-native-fixed32-cap1024-v1',run_id='run',epoch_id='diagnostic-cp10',checkpoint='cp10',remote_job_id='jid',count=32,successes=19,mean_reward=19/32,harness_config={'max_output_tokens':1024},sampling_policy=POLICY['version'],owned_evaluation_policy=POLICY,fixed_task_ids=[str(i)for i in range(32)])
  self.queue=dict(status='complete',request={'label':'label'},records=[self.record]);self.qpath=self.state/'checkpoint-evaluations/q.json';self.save()
  (self.prod/'epoch-first-signed-manifest.json').write_bytes(canonical(self.sign(dict(checkpoint={'id':'cp10'},start=1,epoch='production-20'))))
  self.pointer=self.sign(dict(version='cached1024-dashboard-sources-v1',states=[str(self.state)],indices=list(range(32)),source_sha256='source'))
 def sign(self,v):return dict(payload=v,signer=self.auth,signature=base64.b64encode(self.key.sign(canonical(v)).signature).decode())
 def save(self):self.qpath.write_bytes(canonical(self.queue))
 def project(self):
  with patch('dashboard.cached_evaluator_projection.AUTHORITY',self.auth):return rows(self.pointer,self.prod)
 def test_signed_native_projection_preserves_original_identity(self):
  row=self.project()[0];self.assertEqual((row['successes'],row['epoch_id'],row['original_epoch_id']),(19,'production-20','diagnostic-cp10'));self.assertEqual(json.loads(self.qpath.read_bytes())['records'][0]['epoch_id'],'diagnostic-cp10')
 def test_forged_score_and_wrong_cap_fail(self):
  for field,value in [('successes',20),('sampling_policy','legacy')]:
   old=self.record[field];self.record[field]=value;self.save()
   with self.assertRaises(ValueError):self.project()
   self.record[field]=old
  self.record['harness_config']['max_output_tokens']=128;self.save()
  with self.assertRaises(ValueError):self.project()
 def test_no_ack_or_unmatched_checkpoint_never_invents_zero(self):
  (self.state/'durable-evaluation-acks/jid.json').unlink();self.assertEqual(self.project(),[])
 def test_tampered_durable_ack_and_unauthenticated_pointer_fail(self):
  self.pointer['payload']['source_sha256']='wrong'
  with self.assertRaises(Exception):self.project()
 def add_cap2048_study(self,namespace=True):
  from dashboard.cached_evaluator_projection import CAP2048_PREFIX,CAP2048_EXPERIMENT
  checkpoint='a'*64;epoch=CAP2048_PREFIX+checkpoint[:24]if namespace else'other-unapproved-study'
  old=json.loads((self.state/'durable-evaluation-acks/jid.json').read_bytes())['payload'];ack=copy.deepcopy(old)
  job=ack['original_job']['payload'];manifest=job['manifest']['payload'];manifest.update(epoch=epoch,checkpoint={'id':checkpoint})
  job['manifest']=self.sign(manifest);job['heldout'][0]['harness']['max_output_tokens']=2048;ack['original_job']=self.sign(job)
  raw=canonical(self.sign(ack));(self.state/'durable-evaluation-acks/studyjob.json').write_bytes(raw)
  (self.state/'cache-disposal/studyjob.json').write_text(json.dumps(dict(result={'status':'complete'},durable_ack_sha256=hashlib.sha256(raw).hexdigest())))
  (self.state/'roles/studylabel.json').write_text('{"job_id":"studyjob"}')
  record=dict(self.record,run_id='study',epoch_id=epoch,checkpoint=checkpoint,remote_job_id='studyjob',harness_config={'max_output_tokens':2048},experiment_id=CAP2048_EXPERIMENT)
  (self.state/'checkpoint-evaluations/study.json').write_bytes(canonical(dict(status='complete',request={'label':'studylabel'},records=[record])))
 def test_authorized_separate2048_study_does_not_blank_or_join1024_series(self):
  self.add_cap2048_study();r=self.project();self.assertEqual(len(r),1);self.assertEqual(r[0]['run_id'],'run');self.assertEqual(r[0]['harness_config']['max_output_tokens'],1024)
 def test_unapproved2048_namespace_remains_fail_closed(self):
  self.add_cap2048_study(namespace=False)
  with self.assertRaises(ValueError):self.project()
 def test_ordinary_wrong1024_record_cannot_claim_study_exclusion(self):
  self.record['experiment_id']='owned-cached-native-fixed32-cap2048-v1';self.record['harness_config']['max_output_tokens']=2048;self.save()
  with self.assertRaises(ValueError):self.project()
