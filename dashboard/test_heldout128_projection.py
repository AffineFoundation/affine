import base64,copy,hashlib,io,json,tarfile,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from nacl.signing import SigningKey
from dashboard import heldout128_projection as hp
from dashboard.learner_projection import canonical
class ProjectionTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.prod=self.root/'prod';self.prod.mkdir();self.key=SigningKey.generate();self.auth=self.key.verify_key.encode().hex();self.files={'subnet/model.py':'a'*64};self.acks=[];self.plans=[]
  for g in range(4):
   ids=list(range(6746+32*g,6778+32*g));plan=dict(group=g,env_id='affine_math',indices=ids,seeds=[20261002+i*1000 for i in ids],harness={'version':'text-tools-long-kv-v3','policy':'autoregressive','max_output_tokens':1024,'temperature':.7,'top_p':1.});self.plans.append(plan)
   spec=dict(version='prime-v1-1',id='affine_math',adapter='prime_v1',source_hash='native-hash',config={'dependency_versions':{'grader':'qualified'},'task_snapshot':'assets/math.json'},num_samples=7496,max_turns=1,max_output_tokens=2048,success_reward=1.0)
   manifest=dict(model_runtime_revision='runtime-v1',backend_profile={'dtype':'float32'},numerical_policy={'version':'qualified'},environment_revision='native-v1',harness_source_hash='harness-hash',epoch='research-cp',checkpoint={'id':'cp'},source_bundle={'sha256':'source'},environments=[{'env_id':'affine_math','indices':list(range(6746)),'spec':spec,'harness':{'policy':'native'}}]);job=dict(job_id='original-'+str(g),role='evaluate',manifest=self.sign(manifest),source_files=self.files,runtime_versions={'torch':'qualified'},owned_evaluation_policy=hp.POLICY,heldout=[{k:v for k,v in plan.items()if k!='group'}],created_at=1,expires_at=100)
   report=dict(execution_runtime_revision='runtime-v1',generation_runtime_revision='runtime-v1',backend_profile=manifest['backend_profile'],numerical_policy=manifest['numerical_policy'],job_id=job['job_id'],job_sha256=hp.digest(job),role='evaluate',epoch=manifest['epoch'],checkpoint='cp',source_files=self.files,runtime_versions=job['runtime_versions'],success=True,completed_at=30,chain_transactions=False,heldout_failures=[],heldout=[dict(index=i,seed=seed,reward=int(n<20),classification='positive'if n<20 else'negative',task_hash=str(i),native_graded=True,verified=False,proof_verification_performed=False,trust_scope='operator-owned-process-native-grader',checkpoint='cp')for n,(i,seed)in enumerate(zip(ids,plan['seeds']))]);terminal=dict(job_id=job['job_id'],phase='complete',exit_code=0,started_at=2,finished_at=31)
   objects={'original-job':self.sign(job),'original-report':report,'original-terminal':terminal};a=dict(version='owned-cached-evaluation-durable-ack-v1',group=g,checkpoint=manifest['checkpoint'],original_job=objects['original-job'],original_report=report,original_terminal=terminal,job_sha256=hp.digest(job),report_sha256=hp.digest(report),original_terminal_sha256=hp.digest(terminal),durable_report_full_readback=True,full_readback_objects={k:{'key':'private/never-public','sha256':hp.digest(v),'bytes':len(canonical(v))}for k,v in objects.items()});self.acks.append(self.sign(a))
  self.archive=self.root/'archive.tar.gz';self.receipt=self.root/'archive-ack.json';self.summary=self.root/'summary.json';self.cohort=hp.digest(self.plans);self.scope=dict(version='heldout128-dashboard-sources-v1',source_sha256='source',source_files_sha256=hp.digest(self.files),cohort_sha256=self.cohort,excluded_indices=list(range(7000,7032)),evaluations=[]);self.save()
  (self.prod/'production-first-signed-manifest.json').write_bytes(canonical(self.sign(dict(epoch='production',checkpoint={'id':'cp'},start=1))))
 def sign(self,v):return dict(payload=v,signer=self.auth,signature=base64.b64encode(self.key.sign(canonical(v)).signature).decode())
 def save(self):
  with tarfile.open(self.archive,'w:gz')as t:
   for i,a in enumerate(self.acks):
    raw=canonical(a);m=tarfile.TarInfo('durable-evaluation-acks/'+str(i)+'.json');m.size=len(raw);t.addfile(m,io.BytesIO(raw))
  sha=hashlib.sha256(self.archive.read_bytes()).hexdigest();self.receipt.write_bytes(canonical(dict(R2_full_GET_verified=True,archive_sha256=sha,archive_bytes=self.archive.stat().st_size)))
  score=dict(count=128,successes=80,mean_reward=80/128,cohort_sha256=self.cohort);summary=dict(version='owned-cached-heldout128-checkpoint-actual-v1',cohort_sha256=self.cohort,source_sha256='source',production_checkpoints_changed=False,normal_evaluator_B_paused=False,all_four_genuine_full_R2_ACKs=True,group_owned_model_retired=True,original_jobs=4,task_count=128,checkpoint=self.acks[0]['payload']['checkpoint'],successes=80,completed_at=40,group={'archive_sha256':sha,'result':{'result':{'score':score,'retirement':{'status':'complete'}}}});self.summary.write_bytes(canonical(self.sign(summary)));self.scope['evaluations']=[dict(summary_path=str(self.summary),summary_sha256=hashlib.sha256(self.summary.read_bytes()).hexdigest(),groups={'CP13':dict(archive_path=str(self.archive),archive_ack_path=str(self.receipt),archive_ack_sha256=hashlib.sha256(self.receipt.read_bytes()).hexdigest())})]
 def rows(self):
  with patch.object(hp,'AUTHORITY',self.auth):return hp.rows(self.sign(self.scope),self.prod)
 def test_full128_native_public_allowlist(self):
  row=self.rows()[0];self.assertEqual((row['successes'],row['count'],row['epoch_id']),(80,128,'production'));self.assertFalse(row['proof_verification_performed']);self.assertNotIn('private/',json.dumps(row));self.assertNotIn('archive_path',row)
 def test_partial_missing_summary_never_row(self):
  self.summary.unlink();self.assertEqual(self.rows(),[])
 def test_missing_ack_or_duplicate_group_rejected(self):
  for change in ('missing','duplicate'):
   original=list(self.acks);self.acks=self.acks[:3]if change=='missing'else self.acks[:3]+[self.acks[0]];self.save()
   with self.assertRaises(ValueError):self.rows()
   self.acks=original
 def test_false_native_or_proof_verified_or_bool_reward_rejected(self):
  for field,value in [('native_graded',False),('proof_verification_performed',True),('reward',True)]:
   a=self.acks[0]['payload'];r=a['original_report'];old=r['heldout'][0][field];r['heldout'][0][field]=value;a['report_sha256']=hp.digest(r);a['full_readback_objects']['original-report']={'sha256':hp.digest(r),'bytes':len(canonical(r))};self.acks[0]=self.sign(a);self.save()
   with self.assertRaises(ValueError):self.rows()
   r['heldout'][0][field]=old;a['report_sha256']=hp.digest(r);a['full_readback_objects']['original-report']={'sha256':hp.digest(r),'bytes':len(canonical(r))};self.acks[0]=self.sign(a)
 def test_source_cohort_old32_or_unsigned_rejected(self):
  for key,value in [('source_sha256','wrong'),('cohort_sha256','wrong'),('excluded_indices',[6746])]:
   old=self.scope[key];self.scope[key]=value
   with self.assertRaises(Exception):self.rows()
   self.scope[key]=old
  pointer=self.sign(self.scope);pointer['payload']['source_sha256']='wrong'
  with patch.object(hp,'AUTHORITY',self.auth),self.assertRaises(Exception):hp.rows(pointer,self.prod)
 def test_archive_digest_and_readback_receipt_rejected(self):
  self.archive.write_bytes(self.archive.read_bytes()+b'changed')
  with self.assertRaises(ValueError):self.rows()

 def test_duplicate_paired_checkpoint_originals_rejected(self):
  summary=json.loads(self.summary.read_bytes())['payload'];group=summary['group'];summary=dict(version='owned-cached-heldout128-paired-actual-v1',cohort_sha256=self.cohort,source_sha256='source',production_checkpoints_changed=False,normal_evaluator_B_paused=False,all_eight_genuine_full_R2_ACKs=True,all_group_owned_models_retired=True,original_jobs=8,task_count_per_checkpoint=128,CP11_successes=80,CP12_successes=80,groups={'CP11':group,'CP12':group},completed_at=40)
  self.summary.write_bytes(canonical(self.sign(summary)));entry=self.scope['evaluations'][0];entry['summary_sha256']=hashlib.sha256(self.summary.read_bytes()).hexdigest();paths=entry['groups']['CP13'];entry['groups']={'CP11':paths,'CP12':paths};
  with self.assertRaises(ValueError):self.rows()
 def test_signed_summary_partial_or_fake_retirement_rejected(self):
  for change in ('partial','retirement'):
   summary=json.loads(self.summary.read_bytes())['payload']
   if change=='partial':summary['task_count']=96
   else:summary['group_owned_model_retired']=False
   self.summary.write_bytes(canonical(self.sign(summary)));self.scope['evaluations'][0]['summary_sha256']=hashlib.sha256(self.summary.read_bytes()).hexdigest()
   with self.assertRaises(ValueError):self.rows()
   self.save()

 def test_repeated_scoped_summary_does_not_duplicate_public_row(self):
  self.scope['evaluations'].append(self.scope['evaluations'][0]);self.assertEqual(len(self.rows()),1)

 def rebind(self,group,mutate):
  a=self.acks[group]['payload'];job=a['original_job']['payload'];manifest=job['manifest']['payload'];report=a['original_report'];mutate(job,manifest,report);job['manifest']=self.sign(manifest);a['original_job']=self.sign(job);a['job_sha256']=hp.digest(job);report['job_sha256']=hp.digest(job);a['report_sha256']=hp.digest(report)
  for kind,obj in [('original-job',a['original_job']),('original-report',report)]:a['full_readback_objects'][kind]={'sha256':hp.digest(obj),'bytes':len(canonical(obj))}
  self.acks[group]=self.sign(a);self.save()
 def test_consistent_but_changed_chunk_runtime_rejected(self):
  self.rebind(1,lambda j,m,r:(j['runtime_versions'].update(torch='other'),r['runtime_versions'].update(torch='other')))
  with self.assertRaises(ValueError):self.rows()
 def test_consistent_but_changed_chunk_profile_rejected(self):
  self.rebind(1,lambda j,m,r:(m['backend_profile'].update(dtype='bfloat16'),r['backend_profile'].update(dtype='bfloat16')))
  with self.assertRaises(ValueError):self.rows()
 def test_changed_native_grader_or_snapshot_rejected(self):
  for field in ('dependency_versions','task_snapshot'):
   self.setUp();self.rebind(1,lambda j,m,r:m['environments'][0]['spec']['config'].update({field:'other'}))
   with self.assertRaises(ValueError):self.rows()
 def test_report_profile_does_not_match_original_manifest(self):
  self.rebind(0,lambda j,m,r:r.update(execution_runtime_revision='other'))
  with self.assertRaises(ValueError):self.rows()

 def second_checkpoint(self,change=None):
  first=copy.deepcopy(self.scope['evaluations'][0]);self.acks=copy.deepcopy(self.acks);self.archive=self.root/'archive-other.tar.gz';self.receipt=self.root/'archive-other-ack.json';self.summary=self.root/'summary-other.json'
  for g in range(4):
   def alter(j,m,r):
    m['checkpoint']={'id':'cp2'};j['job_id']='other-'+str(g);r.update(job_id=j['job_id'],checkpoint='cp2')
    for v in r['heldout']:v['checkpoint']='cp2'
    if change:change(j,m,r)
   self.rebind(g,alter);a=self.acks[g]['payload'];a['checkpoint']={'id':'cp2'};a['original_terminal']['job_id']='other-'+str(g);a['original_terminal_sha256']=hp.digest(a['original_terminal']);a['full_readback_objects']['original-terminal']={'sha256':hp.digest(a['original_terminal']),'bytes':len(canonical(a['original_terminal']))};self.acks[g]=self.sign(a)
  self.save();self.scope['evaluations'].insert(0,first);(self.prod/'other-first-signed-manifest.json').write_bytes(canonical(self.sign(dict(epoch='production2',checkpoint={'id':'cp2'},start=2))))
 def test_checkpoint_difference_is_allowed_but_runtime_difference_is_not(self):
  self.second_checkpoint();self.assertEqual(len(self.rows()),2)
  # Every new checkpoint chunk agrees internally, but its runtime differs from
  # the complete original checkpoint. Authentic signatures alone cannot fix it.
  self.setUp();self.second_checkpoint(lambda j,m,r:(j['runtime_versions'].update(torch='other'),r['runtime_versions'].update(torch='other')))
  with self.assertRaises(ValueError):self.rows()
 def test_checkpoint_profile_difference_rejected(self):
  self.second_checkpoint(lambda j,m,r:(m['backend_profile'].update(dtype='bfloat16'),r['backend_profile'].update(dtype='bfloat16')))
  with self.assertRaises(ValueError):self.rows()
