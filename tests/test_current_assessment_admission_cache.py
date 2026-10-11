"""Signed original report fixtures; cache parity and fail-closed adversarial tests."""
import ast,base64,copy,hashlib,importlib,importlib.util,json,math,os,sys,tempfile,unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path
from nacl.signing import SigningKey
from ops import current_assessment_admission_cache as c
import test_continuous_audit_policy as f

def signed(key,p):return dict(payload=p,signer=key.verify_key.encode().hex(),signature=base64.b64encode(key.sign(c.canonical(p)).signature).decode())
def baseline(queue,records,context,authority,cutoff):return c.admit_completed_reports([queue],records,authority,context['workers'],context['approved_sources'],execution_evidence_policy=context['execution_evidence_policy'],cutoff=cutoff,deferral_policy=context['deferral_policy'],historical_report_admission=context['historical_report_admission'])
def output(fn):
 try:return ('return',c.canonical(fn()))
 except Exception as error:return ('error',type(error).__name__,str(error))
def make_policy(directory,root,cache_key,authority):
 pins={str(Path(importlib.import_module(name).__file__).resolve()):hashlib.sha256(Path(importlib.import_module(name).__file__).read_bytes()).hexdigest()for name in c.REQUIRED_MODULES}
 return signed(root,dict(version=c.POLICY_VERSION,admission_authority=authority,directory=str(directory),cache_signer=cache_key.verify_key.encode().hex(),implementation_sha256=hashlib.sha256(Path(c.__file__).read_bytes()).hexdigest(),baseline_module_hashes=pins,maximum_entry_bytes=1024*1024,maximum_directory_bytes=256*1024**2,maximum_entries=10000,minimum_free_bytes=1024**3))
def fixture():
 x=f.PolicyControls();x.setUp();q,pins=x.queue_fixture()
 context=dict(workers={x.worker:['verify']},approved_sources=pins,execution_evidence_policy=None,deferral_policy=None,historical_report_admission=None)
 for k in ('envelope','report','report_request'):q[k]=c.canonical(q[k]).decode()
 return x,q,[x.row],context

def resign_report(x,q,mutate):
 report=json.loads(q['report']);mutate(report);q['report']=c.canonical(report).decode();q['report_digest']=c.digest(report)
 request=json.loads(q['report_request'])['payload'];request['report']=report;q['report_request']=c.canonical(signed(x.key,request)).decode()

class CacheControls(unittest.TestCase):
 def setUp(self):
  self.t=tempfile.TemporaryDirectory();self.path=Path(self.t.name)/'cache';self.path.mkdir(mode=0o700)
  self.x,self.q,self.records,self.context=fixture();self.key=SigningKey.generate();self.policy=make_policy(self.path,self.x.authority,self.key,self.x.root);self.handles=[]
 def tearDown(self):
  for h in self.handles:h.close()
  self.t.cleanup()
 def cache(self,context=None,key=True,policy=None):
  h=c.AdmissionCache(self.policy if policy is None else policy,authority=self.x.root,context=self.context if context is None else context,signing_key=self.key if key else None);self.handles.append(h);return h
 def compare(self,h,q=None,records=None,cutoff=30,context=None):
  q=self.q if q is None else q;records=self.records if records is None else records;context=self.context if context is None else context
  a=output(lambda:baseline(q,records,context,self.x.root,cutoff));b=output(lambda:h.admit(q,records,cutoff=cutoff));self.assertEqual(a,b);return a
 def populate(self):
  h=self.cache();self.assertEqual(self.compare(h)[0],'return');self.assertEqual(self.compare(h,cutoff=3630)[0],'return');self.assertEqual(h.stats['hits'],1);return h,next(self.path.glob('*.json'))
 def test_miss_hit_and_later_hour_exact_original(self):
  h=self.cache();self.compare(h);self.compare(h);self.compare(h,cutoff=3630);self.assertEqual(h.stats,dict(hits=2,misses=1,uncacheable=0,writes=1,baseline_calls=1,rejected_entries=0,write_errors=0,skipped_storage=0))
 def test_readonly_consumer_hit_and_missing_original_fallback(self):
  self.populate();h=self.cache(key=False);self.compare(h);self.assertEqual(h.stats['hits'],1)
  q=dict(self.q,attempt=2);self.compare(h,q=q);self.assertEqual(h.stats['writes'],0)
 def test_all_original_outcomes_match(self):
  h=self.cache()
  for outcome in ('verified_valid','confirmed_invalid','structural_invalid','numerical_ambiguous','infrastructure_error'):
   q=copy.deepcopy(self.q)
   def mutate(report):
    o=report['audits'][0]['outcomes'][0];o.update(fully_audited=outcome!='structural_invalid',valid=True if outcome=='verified_valid'else False if outcome in ('confirmed_invalid','structural_invalid')else None,failure_kind=outcome)
   resign_report(self.x,q,mutate);self.assertEqual(self.compare(h,q=q)[0],'return');self.compare(h,q=q)
  self.assertEqual(h.stats['hits'],5)
 def test_no_error_result_is_cached(self):
  h=self.cache();q=dict(self.q,token='substitution')
  self.assertEqual(self.compare(h,q=q)[0],'error');self.assertEqual(self.compare(h,q=q)[0],'error');self.assertEqual(h.stats['writes'],0);self.assertEqual(h.stats['baseline_calls'],2)
 def test_each_raw_original_byte_change_forces_revalidation(self):
  h=self.cache();self.compare(h)
  for field in ('envelope','report','report_request'):
   q=dict(self.q);q[field]=' '+q[field];self.assertEqual(self.compare(h,q=q)[0],'return')
  self.assertEqual(h.stats['misses'],4);self.assertEqual(h.stats['hits'],0)
 def test_tampered_signatures_refused_despite_original_metadata(self):
  h=self.cache();self.compare(h)
  for field in ('envelope','report_request'):
   q=dict(self.q);doc=json.loads(q[field]);doc['signature']='A'*88;q[field]=c.canonical(doc).decode();self.assertEqual(self.compare(h,q=q)[0],'error')
  self.assertEqual(h.stats['hits'],0)
 def test_report_payload_substitution_with_unchanged_digest_refused(self):
  h=self.cache();self.compare(h);q=dict(self.q);report=json.loads(q['report']);report['audits'][0]['outcomes'][0]['valid']=False;q['report']=c.canonical(report).decode();self.assertEqual(self.compare(h,q=q)[0],'error');self.assertEqual(h.stats['hits'],0)
 def test_every_remaining_queue_field_is_keyed(self):
  h=self.cache();self.compare(h)
  changes={'token':'other','worker':'0'*64,'id':'other','digest':'0'*64,'report_digest':'0'*64,'role':'train','status':'queued','attempt':2,'expires':23.0,'extra':'conservative-new-field'}
  for field,value in changes.items():self.compare(h,q=dict(self.q,**{field:value}))
  self.assertEqual(h.stats['hits'],0);self.assertEqual(h.stats['baseline_calls'],len(changes)+1)
 def test_selected_records_change_revalidated(self):
  h=self.cache();self.compare(h)
  for field in self.records[0]:
   rows=copy.deepcopy(self.records);old=rows[0][field];rows[0][field]=old+1 if type(old)in(int,float)else 'changed'
   self.compare(h,records=rows)
  self.assertEqual(h.stats['hits'],0)
 def test_sources_and_roster_changes_cannot_reuse_entry(self):
  self.populate()
  for change in ('workers','approved_sources','historical_report_admission','deferral_policy'):
   context=copy.deepcopy(self.context)
   if change=='workers':context['workers']={}
   elif change=='approved_sources':context['approved_sources']={}
   elif change=='historical_report_admission':context[change]={'version':'bad','reports':{}}
   else:context[change]='invalid'
   h=self.cache(context=context);self.assertEqual(self.compare(h,context=context)[0],'error');self.assertEqual(h.stats['hits'],0)
 def test_invalid_and_unusual_queue_shapes_use_original(self):
  h=self.cache()
  for field,value in [('envelope',json.loads(self.q['envelope'])),('extra',[]),('extra','\ud800'),('extra',math.nan)]:
   self.compare(h,q=dict(self.q,**{field:value}))
  self.assertEqual(h.stats['uncacheable'],4)
 def execution_context(self):
  source='9'*64;pins=copy.deepcopy(self.context['approved_sources']);pins[source]['subnet/backend_jobs.py']='7'*64
  q=copy.deepcopy(self.q);job=json.loads(q['envelope'])['payload'];job['source_files']=pins[source];q['envelope']=c.canonical(signed(self.x.authority,job)).decode();q['digest']=c.digest(job)
  def mutate(report):report.update(job_sha256=q['digest'],source_files=pins[source],execution_resources_enforced=False)
  resign_report(self.x,q,mutate)
  manifest=job['manifest']['payload'];entry=dict(backend='standard-backend-no-os-resource-enforcement-v1',backend_module_sha256='7'*64,model_runtime_revision=manifest['model_runtime_revision'],backend_profile=manifest['backend_profile'],numerical_policy=manifest['numerical_policy'],runtime_versions=job['runtime_versions'],execution_resources_enforced=False,effective_cutoff=2000)
  context=dict(self.context,approved_sources=pins,execution_evidence_policy=dict(version='explicit-backend-execution-evidence-v2',effective_cutoff=1000,sources={source:entry}),deferral_policy='authenticated-unadmitted-backend-neutral-v1')
  return q,context
 def test_cutoff_activation_boundary_rechecked_on_every_hit(self):
  q,context=self.execution_context();h=self.cache(context=context)
  first=self.compare(h,q=q,context=context,cutoff=1500);self.assertEqual(first[0],'return');self.compare(h,q=q,context=context,cutoff=1999)
  before=baseline(q,self.records,context,self.x.root,1500);after=baseline(q,self.records,context,self.x.root,2000)
  self.assertFalse(before[0]);self.assertTrue(before[1]);self.assertTrue(after[0]);self.assertFalse(after[1])
  self.compare(h,q=q,context=context,cutoff=2000);self.compare(h,q=q,context=context,cutoff=5600);self.compare(h,q=q,context=context,cutoff=1500)
  self.assertEqual(h.stats['hits'],3);self.assertEqual(h.stats['baseline_calls'],2)
 def test_global_activation_and_invalid_cutoffs_use_original_errors(self):
  q,context=self.execution_context();h=self.cache(context=context);self.compare(h,q=q,context=context,cutoff=2000)
  for cutoff in (999,True,None,math.nan,math.inf,-math.inf,'bad'):
   self.compare(h,q=q,context=context,cutoff=cutoff)
  self.assertEqual(h.stats['hits'],0)
 def test_irrelevant_malformed_source_conservatively_uses_original(self):
  q,context=self.execution_context();context['execution_evidence_policy']['sources']['irrelevant']={ 'effective_cutoff':[] }
  h=self.cache(context=context);self.assertEqual(self.compare(h,q=q,context=context,cutoff=2000)[0],'return');self.assertEqual(h.stats['uncacheable'],1)
 def test_returned_mutation_does_not_poison_disk_or_later_hit(self):
  h=self.cache();result=h.admit(self.q,self.records,cutoff=30);result[0].clear();self.compare(h);result=h.admit(self.q,self.records,cutoff=30);result[0].clear();self.compare(h)
 def test_context_is_a_frozen_private_copy(self):
  context=copy.deepcopy(self.context);h=self.cache(context=context);context['workers'].clear();self.compare(h);self.compare(h)
 def test_entry_signature_tampering_rejected(self):
  h,path=self.populate();doc=json.loads(path.read_bytes());doc['payload']['result']['admitted']={};path.write_bytes(c.canonical(doc))
  self.compare(h);self.assertEqual(h.stats['rejected_entries'],1)
 def test_wrong_signed_binding_rejected(self):
  for field in ('key','context_sha256','policy_sha256','version'):
   with self.subTest(field=field):
    h,path=self.populate();doc=json.loads(path.read_bytes());doc['payload'][field]='0'*64;path.write_bytes(c.canonical(signed(self.key,doc['payload'])))
    self.compare(h);self.assertEqual(h.stats['rejected_entries'],1)
    path.unlink()
 def test_wrong_cache_signer_rejected(self):
  h,path=self.populate();payload=json.loads(path.read_bytes())['payload'];path.write_bytes(c.canonical(signed(SigningKey.generate(),payload)))
  self.compare(h);self.assertEqual(h.stats['rejected_entries'],1)
 def test_symlink_cache_entry_rejected(self):
  h,path=self.populate();copy_path=path.with_suffix('.saved');path.rename(copy_path);path.symlink_to(copy_path)
  self.compare(h);self.assertEqual(h.stats['rejected_entries'],1)
 def test_hardlinked_entry_rejected(self):
  h,path=self.populate();os.link(path,path.with_suffix('.linked'))
  self.compare(h);self.assertEqual(h.stats['rejected_entries'],1)
 def test_world_readable_entry_rejected(self):
  h,path=self.populate();path.chmod(0o644)
  self.compare(h);self.assertEqual(h.stats['rejected_entries'],1)
 def test_truncated_entry_rejected(self):
  h,path=self.populate();path.write_bytes(b'{')
  self.compare(h);self.assertEqual(h.stats['rejected_entries'],1)
 def test_oversized_entry_rejected_before_json(self):
  h,path=self.populate();path.write_bytes(b'x'*(1024*1024+1))
  self.compare(h);self.assertEqual(h.stats['rejected_entries'],1)
 def test_replaced_directory_rejected(self):
  h,path=self.populate();self.path.rename(self.path.with_name('old'));self.path.mkdir(mode=0o700)
  self.compare(h);self.assertEqual(h.stats['rejected_entries'],1)
 def test_private_directory_required(self):
  self.path.chmod(0o755)
  with self.assertRaises(ValueError):self.cache()
 def test_directory_symlink_rejected(self):
  other=self.path.with_name('other');self.path.rename(other);self.path.symlink_to(other,target_is_directory=True)
  with self.assertRaises(ValueError):self.cache()
 def test_policy_signature_and_exact_source_pins_required(self):
  for field in ('implementation_sha256','baseline_module_hashes','admission_authority'):
   p=copy.deepcopy(self.policy['payload'])
   if field=='baseline_module_hashes':p[field]={}
   else:p[field]='0'*64
   with self.assertRaises(ValueError):self.cache(policy=signed(self.x.authority,p))
  bad=copy.deepcopy(self.policy);bad['payload']['maximum_entry_bytes']+=1
  with self.assertRaises(Exception):self.cache(policy=bad)
 def test_wrong_private_signing_key_rejected(self):
  with self.assertRaises(ValueError):c.AdmissionCache(self.policy,authority=self.x.root,context=self.context,signing_key=SigningKey.generate())
 def test_read_deadline_never_falls_back_or_is_swallowed(self):
  h,path=self.populate();before=h.stats['baseline_calls']
  with patch.object(c,'authenticate',side_effect=TimeoutError('original evidence deadline')):
   with self.assertRaises(TimeoutError):h.admit(self.q,self.records,cutoff=30)
  self.assertEqual(h.stats['baseline_calls'],before)
 def test_write_deadline_propagates_and_cleans_temporary(self):
  h=self.cache()
  with patch.object(c.os,'fsync',side_effect=TimeoutError('original evidence deadline')):
   with self.assertRaises(TimeoutError):h.admit(self.q,self.records,cutoff=30)
  self.assertFalse(list(self.path.glob('.entry-*')))
 def test_write_io_failure_preserves_original_and_cleans_temporary(self):
  h=self.cache()
  with patch.object(c.os,'fsync',side_effect=OSError('simulated full filesystem')):self.compare(h)
  self.assertEqual(h.stats['write_errors'],1);self.assertFalse(list(self.path.glob('*.json')));self.assertFalse(list(self.path.glob('.entry-*')))
 def test_directory_byte_quota_skips_write_preserving_original(self):
  payload=copy.deepcopy(self.policy['payload']);payload['maximum_directory_bytes']=1024*1024
  # A signed dummy owned file consumes the explicit directory quota.
  occupied=self.path/'occupied';occupied.write_bytes(b'x'*(1024*1024));occupied.chmod(0o600)
  h=self.cache(policy=signed(self.x.authority,payload));self.compare(h);self.assertEqual(h.stats['writes'],0);self.assertEqual(h.stats['skipped_storage'],1)
 def test_entry_count_quota_skips_next_write(self):
  payload=copy.deepcopy(self.policy['payload']);payload['maximum_entries']=1;h=self.cache(policy=signed(self.x.authority,payload));self.compare(h);self.compare(h,q=dict(self.q,attempt=2));self.assertEqual(h.stats['writes'],1);self.assertEqual(h.stats['skipped_storage'],1)
 def test_free_space_reserve_skips_write_preserving_original(self):
  h=self.cache()
  with patch.object(c.os,'fstatvfs',return_value=SimpleNamespace(f_bavail=1024,f_frsize=4096)):self.compare(h)
  self.assertEqual(h.stats['skipped_storage'],1);self.assertFalse(list(self.path.glob('*.json')))
 def test_busy_publication_lock_skips_write(self):
  h=self.cache();lock=os.open(self.path/'.cache-write.lock',os.O_RDWR|os.O_CREAT,0o600)
  try:
   c.fcntl.flock(lock,c.fcntl.LOCK_EX|c.fcntl.LOCK_NB);self.compare(h)
  finally:os.close(lock)
  self.assertEqual(h.stats['skipped_storage'],1)
 def test_source_ast_cutoff_dependencies_have_not_expanded(self):
  module=importlib.import_module('subnet.continuous_audit_policy');tree=ast.parse(Path(module.__file__).read_bytes());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='_admit_report_batch')
  loads=[n for n in ast.walk(fn)if isinstance(n,ast.Name)and n.id=='cutoff'and isinstance(n.ctx,ast.Load)]
  self.assertEqual(len(loads),3)
 def test_downstream_observation_and_snapshot_equal(self):
  h=self.cache();expected=baseline(self.q,self.records,self.context,self.x.root,30)[0];actual=h.admit(self.q,self.records,cutoff=30)[0];again=h.admit(self.q,self.records,cutoff=30)[0]
  for admissions in (actual,again):
   def snapshot(rows):return f.snapshot(self.records,[dict(admitted_queue_job_sha256=k)for k in rows],self.context['workers'],epoch='e1',round=1,checkpoint='a'*64,cutoff=30,audit_policy=self.x.p,admitted_jobs=rows)
   self.assertEqual(c.canonical(snapshot(admissions)),c.canonical(snapshot(expected)))

if __name__=='__main__':unittest.main(verbosity=2)
