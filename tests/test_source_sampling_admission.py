import base64,copy,hashlib,json,shutil,tempfile,unittest
from pathlib import Path
from nacl.signing import SigningKey
from subnet.storage import canonical
from subnet.distributed_roles import Coordinator,digest
from subnet.source_sampling_admission import SamplingAdmission,guarded_coordinator,VERSION
from subnet.backend_profiles import for_config
from subnet.harness import normalize
from subnet.fast_prefill_audit import digest as sha
def sign(k,v):return dict(payload=v,signer=k.verify_key.encode().hex(),signature=base64.b64encode(k.sign(canonical(v)).signature).decode())
class SourceSamplerControls(unittest.TestCase):
 def setUp(self):
  self.key=SigningKey.generate();self.worker=SigningKey.generate();self.authority=self.key.verify_key.encode().hex();self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
  repo=Path(__file__).resolve().parents[1];self.old='0'*63+'1';self.new='0'*63+'2';self.rows={};self.trees={}
  SOURCE_FILES=json.loads((repo/'tests/fixtures/source_sampling_runtime_names.json').read_bytes());self.assertEqual(len(SOURCE_FILES),177)
  for label,s,versions in [('old',self.old,['forced-inverse-cdf-prefill-support-v3']),('new',self.new,['forced-inverse-cdf-prefill-threeway-v4'])]:
   tree=Path(self.tmp.name)/label
   for name in SOURCE_FILES:
    target=tree/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(repo/name,target)
   if label=='old':
    for n in ('forced_sampling.py','fast_prefill_audit.py'):shutil.copyfile(repo/'tests/fixtures/source_sampling_v3'/n,tree/'subnet'/n)
   self.rows[s]=dict(runtime_files={name:hashlib.sha256((tree/name).read_bytes()).hexdigest()for name in SOURCE_FILES},runtime_versions={'torch':'CPU-control-only'},sampling_versions=versions);self.trees[s]=tree
  self.document=dict(version=VERSION,sources=self.rows);self.gate=SamplingAdmission(sign(self.key,self.document),self.authority,self.trees)
  rev,profile,policy=for_config({'model_runtime_revision':'cuda-fp32-eager-sm90-v1'});self.harness={'version':'text-tools-v1','policy':'autoregressive','max_output_tokens':4,'temperature':.7,'top_p':1.}
  self.cal=dict(version='cached-prefill-calibration-v1',checkpoint='a'*64,model_runtime_revision=rev,backend_profile_sha256=sha(profile),harness_sha256=sha(normalize(self.harness)),report_sha256='c'*64,cdf_abs_error=1e-5,logprob_atol=1e-5,toploc_exp_mismatches=0,toploc_mant_err_mean=0,toploc_mant_err_median=0)
  self.manifest=dict(epoch='nonpayable-v4-control',payable=False,checkpoint={'id':'a'*64},source_bundle={'sha256':self.new},model_runtime_revision=rev,backend_profile=profile,numerical_policy=policy,environments=[{'env_id':'tiny','harness':self.harness}],audit_frozen_receipts={'miner':{'sha256':'frozen'}},sampling_source_hash=self.rows[self.new]['runtime_files']['subnet/forced_sampling.py'])
  sampler=self.gate.rows[self.new][2];self.manifest['sampling_contract']=sampler.new_contract(dict(version='forced-inverse-cdf-prefill-threeway-v4',max_attempts=16,calibration=self.cal,uncertainty_adjudication='numerical-inconclusive-no-replay-v1'))
  self.job=dict(schema=1,job_id='v4control',role='verify',created_at=90.,expires_at=200.,manifest=sign(self.key,self.manifest),source_files=self.rows[self.new]['runtime_files'],runtime_versions=self.rows[self.new]['runtime_versions'],submissions=[{'url':'CPU-private-placeholder','sha256':'frozen'}])
 def with_manifest(self,m):return {**self.job,'manifest':sign(self.key,m)}
 def report(self,job=None):
  j=job or self.job;m=j['manifest']['payload'];sampler=self.gate.rows[m['source_bundle']['sha256']][2]
  return dict(job_id=j['job_id'],job_sha256=digest(j),operator=self.authority,role='verify',epoch=m['epoch'],checkpoint=m['checkpoint']['id'],source_files=j['source_files'],runtime_versions=j['runtime_versions'],backend_profile=m['backend_profile'],numerical_policy=m['numerical_policy'],chain_transactions=False,success=True,completed_at=100.,audits=[dict(epoch=m['epoch'],submission_sha256='frozen',sampling_assurance=sampler.assurance(m),accepted=[],outcomes=[dict(batch=0,valid=None,fully_audited=False,failure_kind='numerical_ambiguous',sampling_verification_complete=False,environment_verification_complete=False)])])
 def test_full_signed_enqueue_claim_report_new_v4_positive(self):
  cls=guarded_coordinator(Coordinator,self.gate);q=cls(Path(self.tmp.name)/'queue',self.authority,{self.worker.verify_key.encode().hex():['verify']},clock=lambda:100.)
  q.enqueue(sign(self.key,self.job));claim=q.request(sign(self.worker,dict(action='claim',role='verify',at=100.,nonce='1'*32)))['claim']
  bad=self.report();bad['audits'][0]['outcomes'][0]['valid']=True
  with self.assertRaises(ValueError):q.request(sign(self.worker,dict(action='report',at=100.,nonce='3'*32,job_id=self.job['job_id'],token=claim['token'],report=bad)))
  self.assertEqual(q.status(self.job['job_id'])['status'],'leased')
  response=q.request(sign(self.worker,dict(action='report',at=100.,nonce='2'*32,job_id=self.job['job_id'],token=claim['token'],report=self.report())))
  self.assertTrue(response['accepted']);self.assertEqual(q.status(self.job['job_id'])['status'],'complete')
 def test_old_v3_original_source_remains_original_sampler(self):
  m=copy.deepcopy(self.manifest);sampler=self.gate.rows[self.old][2];m['source_bundle']['sha256']=self.old;m['sampling_source_hash']=self.rows[self.old]['runtime_files']['subnet/forced_sampling.py'];m['sampling_contract']=sampler.new_contract(dict(version='forced-inverse-cdf-prefill-support-v3',max_attempts=16,calibration=self.cal,support_adjudication='exact-cached-replay-v1'));j=self.with_manifest(m);j['source_files']=self.rows[self.old]['runtime_files'];self.gate.check(j);self.gate.report(j,self.report(j))
 def test_unknown_version_oldsource_rebinding_wronghash_runtime_refused(self):
  for change in ('oldsource','version','sampler','runtime','map','foreignsource','calibration','environment'):
   j=copy.deepcopy(self.job);m=j['manifest']['payload']
   if change=='oldsource':m['source_bundle']['sha256']=self.old;j['source_files']=self.rows[self.old]['runtime_files'];m['sampling_source_hash']=self.rows[self.old]['runtime_files']['subnet/forced_sampling.py']
   if change=='version':m['sampling_contract']['version']='unapproved-v9'
   if change=='sampler':m['sampling_source_hash']='0'*64
   if change=='runtime':j['runtime_versions']={'torch':'foreign'}
   if change=='map':j['source_files']['subnet/forced_sampling.py']='0'*64
   if change=='foreignsource':m['source_bundle']['sha256']='0'*64
   if change=='environment':m['environments']=[]
   if change=='calibration':m['sampling_contract']['calibration']['checkpoint']='b'*64
   j['manifest']=sign(self.key,m)
   with self.subTest(change=change),self.assertRaises(ValueError):self.gate.check(j)
 def test_report_assurance_unknown_falsecredit_and_receipt_mutation_refused(self):
  for change in ('assurance','valid','fully','sampling','environment','accepted','receipt','positions'):
   r=self.report();a=r['audits'][0];o=a['outcomes'][0]
   if change=='assurance':a['sampling_assurance']['version']='forced-inverse-cdf-prefill-support-v3'
   if change=='valid':o['valid']=True
   if change=='fully':o['fully_audited']=True
   if change=='sampling':o['sampling_verification_complete']=True
   if change=='environment':o['environment_verification_complete']=True
   if change=='positions':o.update(uncertain_token_positions=[-1],uncertain_token_position_count=-1)
   if change=='receipt':a['outcomes']=[];a['accepted']=[dict(index=2,env_id='tiny',rollouts=[dict(seed=0,sampling={'foreign':'wrong'})])]
   if change=='accepted':a['accepted']=[dict(env_id=None,index=None,rollouts=[])]
   with self.subTest(change=change),self.assertRaises(ValueError):self.gate.report(self.job,r)
 def test_registry_signature_runtime_map_and_unsupported_original_version_refused(self):
  for change in ('signature','hash','versions','missing'):
   d=copy.deepcopy(self.document)
   if change=='hash':d['sources'][self.new]['runtime_files']['subnet/model.py']='0'*64
   if change=='versions':d['sources'][self.old]['sampling_versions']=['forced-inverse-cdf-prefill-threeway-v4']
   if change=='missing':d['sources'][self.new]['runtime_files'].pop('subnet/model.py')
   e=sign(self.key,d)
   if change=='signature':e['payload']['version']='foreign'
   with self.subTest(change=change),self.assertRaises(Exception):SamplingAdmission(e,self.authority,self.trees)
